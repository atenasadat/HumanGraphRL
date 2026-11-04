"""LoRA on-policy GRPO with causal potential-based control variates.

Phi(s_T)=Phi(s_0)=0. Undiscounted episodic gamma=1 matches the task objective.
The shaping return-to-go is -Phi(s_t), a prefix-only baseline with zero expected
score-function gradient. It cannot supply task information on all-failure groups.
A lagged gradient-norm controller targets a 35% sampled component norm share;
this is NOT a guaranteed 35% useful gradient or proof of policy invariance of GRPO.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'data').is_dir())
HERE = ROOT/'processed_data/cognitive_priors_and_audits_from_transcripts'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from scripts.experiment.thinkaloud_rl.reasoning_shaper import load_blueprint, compute_trajectory_shaping, classify_reasoning_step, segment_thought_stream
from scripts.experiment.thinkaloud_rl.multidomain_tasks import SYSTEM, select_tasks, outcome_reward
from scripts.experiment.thinkaloud_rl.benchmark_loader import (
    load_livecodebench, load_math500, load_aime, load_wildbench_writing,
    evaluate_lcb_sample, evaluate_math500_sample, evaluate_aime_sample, evaluate_wildbench_sample
)
from scripts.experiment.thinkaloud_rl.training_utils import causal_potentials, potential_rewards, discounted_sum, normalized_advantages, policy_loss, grad_norm, calibrated_scale



def extract_cognitive_shaping_reward(solution_str, gamma=.99):
    """Compatibility helper: invariant whole-trajectory shaping return is zero."""
    import re
    match = re.search(r'<think>(.*?)</think>',solution_str,re.S)
    moves = [classify_reasoning_step(s) for s in segment_thought_stream(match[1])] if match else []
    return discounted_sum(compute_trajectory_shaping(moves,gamma=gamma),gamma)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--shaping-weight',type=float,default=1.)
    p.add_argument('--condition',choices=['baseline','domain_specific','unified'],default='domain_specific')
    p.add_argument('--train-domain',choices=['programming','writing','mixed'],default='mixed')
    p.add_argument('--blueprint',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'pilots/results/legacy_runs/multidomain_grpo')
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--target-shape-share',type=float,default=.35)
    p.add_argument('--potential-scale',type=float,default=.1)
    p.add_argument('--max-potential-scale',type=float,default=10.)
    p.add_argument('--fixed-scale',action='store_true')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--load-in-4bit',action='store_true',help='NF4 base weights with FP16 computation for the mixed-GPU suite')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--benchmark',choices=['none','livecodebench','math500','aime','wildbench','multidomain_bench'],default='none',help='Train on 2024-2026 gold-standard benchmarks')
    p.add_argument('--benchmark-dir',type=Path,default=ROOT/'data/benchmarks',help='Directory with downloaded benchmarks on CARC')
    p.add_argument('--dry-run',action='store_true')
    return p


def choose_device(requested='auto'):
    return requested if requested!='auto' else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_model(model_name, device, local_only=False, adapter=None, load_in_4bit=False):
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(model_name,local_files_only=local_only)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    if load_in_4bit:
        if device != 'cuda': raise ValueError('The mixed-GPU NF4 configuration requires CUDA')
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                       bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,
            quantization_config=quantization,device_map={'':torch.cuda.current_device()},
            attn_implementation='eager',local_files_only=local_only)
        # Apply the same fp32 non-quantized-layer casts on training and reload.
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    else:
        model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype,local_files_only=local_only).to(device)
    if adapter: model=PeftModel.from_pretrained(model,str(adapter))
    return model,tok


def generate(model,tok,task,device,max_tokens,sample=True):
    rendered=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':task['prompt']}],tokenize=False,add_generation_prompt=True)
    inputs=tok(rendered,return_tensors='pt').to(device)
    with torch.no_grad():
        kwargs={'do_sample':sample}
        if sample: kwargs.update(temperature=1.,top_p=1.,top_k=0)
        output=model.generate(**inputs,max_new_tokens=max_tokens,pad_token_id=tok.pad_token_id,**kwargs)
    prefix=inputs['input_ids'].shape[1]
    ids=output[0,prefix:].tolist()
    return output,prefix,ids,tok.decode(ids,skip_special_tokens=True)


def evaluate_task_outcome(text, task):
    if 'test_cases' in task:
        return 1.0 if evaluate_lcb_sample(text, task) else 0.0
    if 'target_answer' in task:
        return 1.0 if evaluate_math500_sample(text, task) else 0.0
    if 'target_integer' in task:
        return 1.0 if evaluate_aime_sample(text, task) else 0.0
    if task.get('domain') == 'writing' and 'concepts' not in task:
        return evaluate_wildbench_sample(text)
    return outcome_reward(text, task)


def main():
    args=parser().parse_args()
    if args.steps < 1 or args.group_size < 2 or args.max_new_tokens < 2: raise ValueError('Positive steps, group-size >=2, max-new-tokens >=2 required')
    if not .3 <= args.target_shape_share <= .4: raise ValueError('Target share must be between .3 and .4')
    if any(not math.isfinite(v) or v<0 for v in (args.shaping_weight,args.potential_scale,args.max_potential_scale)): raise ValueError('Scales must be finite and nonnegative')
    
    if args.benchmark == 'livecodebench':
        tasks = load_livecodebench(args.benchmark_dir)
    elif args.benchmark == 'math500':
        tasks = load_math500(args.benchmark_dir)
    elif args.benchmark == 'aime':
        tasks = load_aime(args.benchmark_dir)
    elif args.benchmark == 'wildbench':
        tasks = load_wildbench_writing(args.benchmark_dir)
    elif args.benchmark == 'multidomain_bench':
        lcb_tasks = load_livecodebench(args.benchmark_dir, limit=100)
        write_tasks = load_wildbench_writing(args.benchmark_dir)
        tasks = lcb_tasks + write_tasks
    else:
        tasks = select_tasks(args.train_domain,'train')
    weight=0. if args.condition=='baseline' else args.shaping_weight
    blueprints={}
    for domain in ('programming','writing'):
        name='unified' if args.condition=='unified' else 'proposal' if domain=='writing' else 'programming'
        path=args.blueprint or HERE/f'{name}_cognitive_blueprint.json'
        blueprints[domain]=load_blueprint(path)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(effective_shaping_weight=weight, gamma=1., lora_rank=16,lora_alpha=32,
                  targets=['q_proj','k_proj','v_proj','o_proj'],reward='heuristic_rubric_not_task_success',
                  shape_share_definition='norm(g_shape)/(norm(g_task)+norm(g_shape)); lagged calibration',
                  task_ids=[t['id'] for t in tasks])
    if args.dry_run:
        print(json.dumps(config,indent=2)); return
    if (args.output_dir/'config.json').exists(): raise FileExistsError(f'Refusing to overwrite run: {args.output_dir}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed); rng=random.Random(args.seed)
    device=choose_device(args.device)
    model,tok=load_model(args.model,device,args.local_files_only,load_in_4bit=args.load_in_4bit)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type=TaskType.CAUSAL_LM,target_modules=config['targets']))
    model.print_trainable_parameters()
    # eval() disables dropout but does NOT disable gradients, ensuring rollout
    # and likelihood policies match. Each rollout is backpropagated separately.
    model.eval()
    if args.load_in_4bit:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # Checkpointing requires train mode. Disable dropout explicitly so the
        # rollout and likelihood policies still match.
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout): module.p=0.
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    config.update(device=device, trainable_parameters=sum(p.numel() for p in params), torch_version=torch.__version__,
                  gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                  gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None)
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for domain,bp in blueprints.items(): (args.output_dir/f'{domain}_blueprint.json').write_text(json.dumps(bp,indent=2)+'\n')
    scales={d:args.potential_scale for d in blueprints}
    history=[]
    with (args.output_dir/'rollouts.jsonl').open('w') as rollout_file, (args.output_dir/'metrics.jsonl').open('w') as metrics_file:
        for step in range(1,args.steps+1):
            started=time.monotonic(); task=rng.choice(tasks); domain=task['domain']; bp=blueprints[domain]
            model.eval()
            rollouts=[generate(model,tok,task,device,args.max_new_tokens) for _ in range(args.group_size)]
            if args.load_in_4bit: model.train()
            rewards=[evaluate_task_outcome(r[3],task) for r in rollouts]
            advantages=normalized_advantages(rewards)
            task_grads=[torch.zeros_like(p) for p in params]; shape_grads=[torch.zeros_like(p) for p in params]
            total_task_loss=total_shape_loss=0.; max_residual=0.
            for index,((ids,prefix,generated,text),adv,reward) in enumerate(zip(rollouts,advantages,rewards)):
                phis=causal_potentials(tok,generated,bp,args.max_new_tokens) if weight else [0.]*(len(generated)+1)
                residual=abs(discounted_sum(potential_rewards(phis)))
                max_residual=max(max_residual,residual)
                if residual > 1e-8: raise ArithmeticError('Telescoping invariant failed')
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,prefix-1:-1].float()
                logp=logits.log_softmax(-1).gather(-1,ids[0,prefix:].unsqueeze(-1)).squeeze(-1)
                task_loss=policy_loss(logp,adv.to(device),args.max_new_tokens)/args.group_size
                tg=torch.autograd.grad(task_loss,params,retain_graph=bool(weight))
                for acc,g in zip(task_grads,tg): acc.add_(g.detach())
                total_task_loss+=float(task_loss.detach())
                if weight:
                    # Exact gamma=1 shaped return-to-go, not immediate reward or
                    # a non-invariant mean of shaping increments.
                    shape_adv=-torch.tensor(phis[:-1],device=device,dtype=logp.dtype)
                    shape_loss=policy_loss(logp,shape_adv,args.max_new_tokens)/args.group_size
                    sg=torch.autograd.grad(shape_loss,params)
                    for acc,g in zip(shape_grads,sg): acc.add_(g.detach())
                    total_shape_loss+=float(shape_loss.detach())
                rollout_file.write(json.dumps({'step':step,'sample':index,'task_id':task['id'],'domain':domain,'text':text,'token_ids':generated,'task_reward':reward,'shaping_discounted_return':residual})+'\n')
            task_norm,shape_norm=grad_norm(task_grads),grad_norm(shape_grads)
            scale=scales[domain]*weight
            # Keep the scale determined by PRIOR groups. Gating the control
            # variate on this group's outcomes would spoil its zero-mean property.
            optimizer.zero_grad(set_to_none=True)
            for p,gt,gs in zip(params,task_grads,shape_grads): p.grad=gt+scale*gs
            combined_norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if combined_norm > 0:
                optimizer.step()
            share=scale*shape_norm/(task_norm+scale*shape_norm) if task_norm+scale*shape_norm else 0.
            cosine=sum(float((gt.float()*gs.float()).sum()) for gt,gs in zip(task_grads,shape_grads))/(task_norm*shape_norm) if task_norm*shape_norm else None
            row={'step':step,'domain':domain,'task_id':task['id'],'mean_task':sum(rewards)/len(rewards),
                 'task_grad_norm':task_norm,'unscaled_shape_grad_norm':shape_norm,'potential_scale_used':scale,
                 'shape_gradient_norm_share':share,'task_shape_cosine':cosine,'combined_grad_norm':combined_norm,
                 'no_task_gradient':task_norm<=1e-12,'max_telescoping_residual':max_residual,
                 'control_variate_only_update':task_norm<=1e-12 and scale*shape_norm>0,
                 'loss':total_task_loss+scale*total_shape_loss,'time_sec':time.monotonic()-started}
            if weight and not args.fixed_scale:
                scales[domain]=calibrated_scale(task_norm,shape_norm,args.target_shape_share,args.max_potential_scale)/weight
            history.append(row); metrics_file.write(json.dumps(row)+'\n'); metrics_file.flush(); rollout_file.flush()
            print(json.dumps(row),flush=True)
    model.save_pretrained(args.output_dir/'adapter'); tok.save_pretrained(args.output_dir/'adapter')
    (args.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')

if __name__=='__main__': main()
