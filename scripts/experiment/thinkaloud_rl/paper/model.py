"""Consistent policy loading, sampling, and logits for training and evaluation."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

SYSTEM='Solve the task. Put your reasoning inside <think>...</think>, then give the final answer.'


def load(config, adapter=None, trainable=False):
    device=config.get('device','cuda')
    revision=config.get('model_revision')
    tok=AutoTokenizer.from_pretrained(config['model'],revision=revision,local_files_only=config.get('offline',True))
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    kwargs={'revision':revision,'local_files_only':config.get('offline',True),'attn_implementation':'eager'}
    if config.get('load_in_4bit',True):
        if device!='cuda':raise ValueError('NF4 requires CUDA; set load_in_4bit=false for CPU smoke tests')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),device_map={'':torch.cuda.current_device()})
    else:kwargs['torch_dtype']=torch.float32 if device=='cpu' else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
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
    started=time.monotonic()
    rendered=render(tok,task['prompt'],config.get('system_prompt'))
    inp=tok(rendered,return_tensors='pt').to(next(model.parameters()).device)
    if inp['input_ids'].shape[1]>config.get('max_prompt_tokens',8192):
        raise ValueError('Prompt exceeds configured budget; no silent truncation: '+task.get('id','judge'))
    horizon=config['max_new_tokens']
    constraints={}
    if config.get('response_json_schema') is not None:
        from .constrained import json_prefix_filter
        constraints={'prefix_allowed_tokens_fn':json_prefix_filter(tok,config['response_json_schema']),
                     'forced_eos_token_id':None}
    model.eval()
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=horizon,do_sample=not greedy,
            **({} if greedy else {'temperature':1.,'top_p':1.,'top_k':0}),
            pad_token_id=tok.pad_token_id,**constraints)
    prefix=inp['input_ids'].shape[1]
    ids=out[0,prefix:].tolist()
    from .telemetry import generation
    generation(config,prefix,len(ids),time.monotonic()-started)
    raw=tok.decode(ids,skip_special_tokens=True)
    opens=rendered.rfind('<think>')>rendered.rfind('</think>')
    text=('<think>' if opens else '')+raw
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    return {'sequence':out,'prefix':prefix,'ids':ids,'text':text,'raw_text':raw,
            'prompt_opens_think':opens,'truncated':bool(ids) and len(ids)>=horizon and ids[-1] not in eos}


def log_probs(model,rollout):
    model.train()
    ids=rollout['sequence']
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,rollout['prefix']-1:-1].float()
    logp=logits.log_softmax(-1)
    selected=logp.gather(-1,ids[0,rollout['prefix']:].unsqueeze(-1)).squeeze(-1)
    return selected,logp
