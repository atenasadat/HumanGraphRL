"""Consistent policy loading, sampling, and logits for training and evaluation."""
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

SYSTEM=('Solve the task. Inside <think>...</think>, give a concise analysis of the requirements, '
        'your approach and any checks. Close </think> before writing the final answer or requested artifact. '
        'The reasoning section should explain your decisions; put the requested artifact after it.')
REASONING_SYSTEM=('Analyze the task and explain your reasoning concisely, including your decisions '
                  'and any checks you actually make. Do not write the final deliverable yet. '
                  'Use plain text, without XML tags.')
ANSWER_SYSTEM=('Complete the requested task using the analysis as a draft, correcting any mistakes. '
               'Return only the final deliverable, without a reasoning section or XML tags. '
               'For programming, return the complete Python implementation in one python code block.')


class StopOnSignal(StoppingCriteria):
    def __call__(self,input_ids,scores,**kwargs):
        from .train import STOP
        return torch.full((input_ids.shape[0],),STOP,device=input_ids.device,dtype=torch.bool)


def load(config, adapter=None, trainable=False):
    device=config.get('device','cuda')
    revision=config.get('model_revision')
    tok=AutoTokenizer.from_pretrained(config['model'],revision=revision,local_files_only=config.get('offline',True))
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    attention=config.get('attention_implementation','eager')
    if attention not in ('eager','sdpa'):raise ValueError('Unsupported attention implementation: '+str(attention))
    kwargs={'revision':revision,'local_files_only':config.get('offline',True),'attn_implementation':attention}
    if config.get('load_in_4bit',True):
        if device!='cuda':raise ValueError('NF4 requires CUDA; set load_in_4bit=false for CPU smoke tests')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),device_map={'':torch.cuda.current_device()})
    else:
        default_dtype='float32' if device in ('cpu','mps') else 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
        dtype=config.get('dtype',default_dtype)
        if dtype not in ('float32','float16','bfloat16'):raise ValueError('Unsupported model dtype: '+str(dtype))
        # torch_dtype also works with the older Transformers versions allowed
        # by requirements-paper.txt; dtype was introduced later.
        kwargs['torch_dtype']=getattr(torch,dtype)
    model=AutoModelForCausalLM.from_pretrained(config['model'],**kwargs)
    if config.get('load_in_4bit',True):
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    else:model=model.to(device)
    if adapter:
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable)
    elif trainable:
        model=get_peft_model(model,LoraConfig(r=config.get('lora_rank',16),lora_alpha=32,lora_dropout=0.,
            bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj']))
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    if trainable and config.get('gradient_checkpointing',True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.eval()
    return model,tok


def render(tok,prompt,system=None):
    # Disable auto-thinking where supported; our explicit protocol stays identical.
    from jinja2.exceptions import TemplateError
    try:
        return tok.apply_chat_template([{'role':'system','content':system or SYSTEM},{'role':'user','content':prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TemplateError:
        # Models without a system role receive the identical instruction in the user turn.
        return tok.apply_chat_template([{'role':'user','content':(system or SYSTEM)+'\n\n'+prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)


def sample(model,tok,task,config,greedy=False):
    protocol=config.get('generation_protocol','tagged_single_pass')
    if protocol not in ('tagged_single_pass','two_stage_v1'):
        raise ValueError('Unknown generation protocol: '+str(protocol))
    if protocol=='two_stage_v1' and not config.get('system_prompt') and not config.get('response_json_schema'):
        horizon=config['max_new_tokens'];fraction=config.get('reasoning_fraction',.25)
        if not 0<fraction<1 or horizon<2:raise ValueError('Invalid two-stage token budget')
        reasoning_budget=max(1,min(horizon-1,int(horizon*fraction)))
        reasoning=_sample(model,tok,task,{**config,'system_prompt':REASONING_SYSTEM,
                          'max_new_tokens':reasoning_budget},greedy)
        from .train import STOP
        if STOP:raise InterruptedError
        draft=reasoning['raw_text']
        final_task={**task,'prompt':'Original task:\n'+task['prompt']+
                    '\n\nDraft analysis (may contain mistakes):\n'+draft+
                    '\n\nNow provide the final deliverable for the original task.'}
        answer=_sample(model,tok,final_task,{**config,'system_prompt':ANSWER_SYSTEM,
                       'max_new_tokens':horizon-reasoning_budget},greedy)
        # Delimiters are serialization, not sampled reasoning or trainable actions.
        escaped=draft.replace('<think>','&lt;think&gt;').replace('</think>','&lt;/think&gt;')
        text='<think>'+escaped+'</think>\n'+answer['raw_text']
        return {'segments':[reasoning,answer],'ids':reasoning['ids']+answer['ids'],
                'prefix':reasoning['prefix']+answer['prefix'],'text':text,'raw_text':text,
                'prompt_opens_think':True,'reasoning_tokens':len(reasoning['ids']),
                'generation_protocol':protocol,'reasoning_truncated':reasoning['truncated'],
                'answer_truncated':answer['truncated'],
                'truncated':reasoning['truncated'] or answer['truncated']}
    return _sample(model,tok,task,config,greedy)


def _sample(model,tok,task,config,greedy=False):
    started=time.monotonic()
    rendered=render(tok,task['prompt'],config.get('system_prompt'))
    if config.get('reasoning_prefill',True) and not config.get('system_prompt'):
        rendered+='<think>\n'
    inp=tok(rendered,return_tensors='pt').to(next(model.parameters()).device)
    if inp['input_ids'].shape[1]>config.get('max_prompt_tokens',8192):
        raise ValueError('Prompt exceeds configured budget; no silent truncation: '+task.get('id','judge'))
    horizon=config['max_new_tokens']
    temperature=config.get('sampling_temperature',1.)
    if not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Sampling temperature must be finite and positive')
    constraints={}
    if config.get('response_json_schema') is not None:
        from .constrained import json_prefix_filter
        constraints={'prefix_allowed_tokens_fn':json_prefix_filter(tok,config['response_json_schema'])}
    model.eval()
    # Model presets can silently add repetition penalties or other logit warpers.
    # Use only the declared temperature, matching log_probs() exactly.
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=horizon,do_sample=not greedy,
            **({} if greedy else {'temperature':temperature,'top_p':1.,'top_k':0}),
            repetition_penalty=1.,encoder_repetition_penalty=1.,no_repeat_ngram_size=0,
            num_beams=1,typical_p=1.,min_p=None,epsilon_cutoff=0.,eta_cutoff=0.,
            bad_words_ids=None,suppress_tokens=None,begin_suppress_tokens=None,
            forced_bos_token_id=None,forced_eos_token_id=None,min_length=0,min_new_tokens=None,
            sequence_bias=None,renormalize_logits=False,
            stopping_criteria=StoppingCriteriaList([StopOnSignal()]),
            pad_token_id=tok.pad_token_id,**constraints)
    prefix=inp['input_ids'].shape[1]
    ids=out[0,prefix:].tolist()
    from .telemetry import generation
    generation(config,prefix,len(ids),time.monotonic()-started)
    from .train import STOP
    if STOP:raise InterruptedError
    raw=tok.decode(ids,skip_special_tokens=True)
    opens=rendered.rstrip().endswith('<think>')
    text=('<think>' if opens else '')+raw
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    return {'sequence':out,'prefix':prefix,'ids':ids,'text':text,'raw_text':raw,
            'sampling_temperature':temperature,
            'prompt_opens_think':opens,'truncated':bool(ids) and len(ids)>=horizon and ids[-1] not in eos}


def log_probs(model,rollout):
    if 'segments' in rollout:
        values=[log_probs(model,part) for part in rollout['segments']]
        return torch.cat([v[0] for v in values]),torch.cat([v[1] for v in values])
    model.train()
    ids=rollout['sequence']
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,rollout['prefix']-1:-1].float()
    logp=(logits/rollout.get('sampling_temperature',1.)).log_softmax(-1)
    selected=logp.gather(-1,ids[0,rollout['prefix']:].unsqueeze(-1)).squeeze(-1)
    return selected,logp
