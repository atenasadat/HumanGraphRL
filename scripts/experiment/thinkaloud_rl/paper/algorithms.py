"""Explicit single-update estimators and paper-derived reward controls."""
import re
import torch
from ..reasoning_shaper import PURE_MOVES, classify_reasoning_step, segment_thought_stream, compute_move_potential

METHODS=('grpo','grpo_sequence','dr_grpo','dapo_one_update','rloo','shrinkage','t2t','dra','entropy','human','shuffled',
         'uniform','frequency','transition','phase_free','synthetic','learned_prefix')
COGNITIVE={'human','shuffled','uniform','frequency','transition','phase_free','synthetic'}
from .structure import TOPOLOGY
METHODS=METHODS+tuple(sorted(TOPOLOGY))


def advantages(rewards, method='grpo'):
    """Rewards: [prompts, samples]. Shrinkage: arXiv:2511.03710v2, Eqs 13–16."""
    r=torch.as_tensor(rewards,dtype=torch.float32)
    if r.ndim!=2 or r.shape[1]<2:
        raise ValueError('Expected prompts x group_size, group_size >= 2')
    n,m=r.shape
    mean=r.mean(1,keepdim=True)
    loo=(r.sum(1,keepdim=True)-r)/(m-1)
    if method=='rloo':
        return r-loo
    if method=='shrinkage':
        if n<2:
            raise ValueError('Shrinkage requires at least two independent prompts per batch')
        variance=((r-mean)**2).sum(1)/(m*(m-1))
        baselines=[]
        for i in range(n):
            others=torch.arange(n)!=i
            global_mean=mean[others].mean()
            v=variance[others].mean()
            s=((mean[others]-global_mean)**2).mean()
            lam=(n-1)/n*v/(v+s).clamp_min(1e-12)
            baselines.append((1-lam)*loo[i]+lam*global_mean)
        return r-torch.stack(baselines)
    centered=r-mean
    return centered if method=='dr_grpo' else centered/r.std(1,unbiased=False,keepdim=True).clamp_min(1e-8)


def t2t_rewards(rewards, lengths, horizon, alpha=.2, min_length=0):
    """T2T arXiv:2602.04265v4 Eq 13. Same optimizer as controlled GRPO."""
    r=torch.as_tensor(rewards,dtype=torch.float32)
    if not torch.all((r==0)|(r==1)):
        raise ValueError('T2T requires binary correctness rewards; writing proxies are unsupported')
    if not 0<alpha<.5 or horizon<=min_length:
        raise ValueError('Invalid T2T alpha or length range')
    length=((torch.as_tensor(lengths,dtype=torch.float32)-min_length)/(horizon-min_length)).clamp(0,1)
    p=r.mean(1,keepdim=True)
    return r*(1-alpha*length*p)+(1-r)*alpha*length*(1-p)


def dra_rewards(rewards, embeddings):
    """DRA ACL 2026 Algorithm 1: inverse cosine-similarity row-sum weighting.

    Uses the flattened prompt batch, as in the published pseudocode. Abort on
    nonpositive density instead of silently substituting a different kernel.
    """
    r=torch.as_tensor(rewards,dtype=torch.float32)
    e=torch.nn.functional.normalize(torch.as_tensor(embeddings,dtype=torch.float32),dim=-1)
    if e.shape[0]!=r.numel():raise ValueError('Embedding/reward count mismatch')
    density=(e@e.T).sum(1)+1e-6
    if not torch.isfinite(density).all() or (density<=0).any():
        raise ValueError('DRA cosine density is nonpositive; inspect embeddings')
    return r/density.reshape(r.shape)


def dapo_soft_overlong(reward,length,horizon,buffer):
    """DAPO soft length penalty; explicit budget-truncation convention in README."""
    if not 0<buffer<=horizon:raise ValueError('Invalid DAPO length buffer')
    return reward+min(0.,(horizon-buffer-length)/buffer)


def prefix_arrays(tokenizer, ids, prior, horizon, prompt_opens_think=False, reasoning_tokens=None):
    """Before-action potentials and value features; never inspect future tokens.

    Incrementally tokenize the last thought fragment, caching classifications of
    completed segments. Potentials use within-budget token position, documented
    separately from the human within-session event phase.
    """
    phis=[]; features=[]; cache={}
    for t in range(len(ids)):
        if reasoning_tokens is not None:
            # Only past reasoning tokens; the deterministic handoff adds no action.
            draft=tokenizer.decode(ids[:min(t,reasoning_tokens)],skip_special_tokens=True)
            draft=draft.replace('<think>','&lt;think&gt;').replace('</think>','&lt;/think&gt;')
            prefix='<think>'+draft+('</think>' if t>=reasoning_tokens else '')
        else:
            prefix=tokenizer.decode(ids[:t],skip_special_tokens=True)
        if reasoning_tokens is None and prompt_opens_think:
            prefix='<think>'+prefix
        match=re.search(r'<think>(.*?)(?:</think>|$)',prefix,re.S)
        units=segment_thought_stream(match[1]) if match else []
        moves=[]
        for u in units:
            if u not in cache:
                cache[u]=classify_reasoning_step(u)
            moves.append(cache[u])
        count=max(1,len(moves))
        features.append([t/horizon,*[moves.count(m)/count for m in PURE_MOVES],float(bool(match)),float('</think>' in prefix)])
        phase='early' if t/horizon<.33 else 'mid' if t/horizon<.66 else 'late'
        phi=compute_move_potential(moves,horizon,blueprint=prior,
            alpha=prior.get('potential_alpha',.5),beta=prior.get('potential_beta',.5),phase_override=phase) if moves and t else 0.
        phis.append(phi)
    return phis,features


class GradientMoments:
    """Exact trace variance/covariance across independent rollout groups."""
    def __init__(self):
        self.n=0;self.x=None;self.y=None;self.xx=0.;self.yy=0.;self.xy=0.

    def add(self,x,y):
        x=x.detach().float().cpu(); y=y.detach().float().cpu()
        if self.x is None:
            self.x=torch.zeros_like(x);self.y=torch.zeros_like(y)
        self.n+=1;self.x+=x;self.y+=y
        self.xx+=float(x.double().square().sum());self.yy+=float(y.double().square().sum())
        self.xy+=float((x.double()*y.double()).sum())

    def report(self,scale):
        if self.n<2:
            raise ValueError('At least two independent groups required')
        vx=max(0.,(self.xx-float(self.x.double().square().sum())/self.n)/(self.n-1))
        vy=max(0.,(self.yy-float(self.y.double().square().sum())/self.n)/(self.n-1))
        cov=(self.xy-float((self.x.double()*self.y.double()).sum())/self.n)/(self.n-1)
        combined=max(0.,vx+2*scale*cov+scale*scale*vy)
        return {'groups':self.n,'task_variance_trace':vx,'shape_variance_trace':vy,'covariance_trace':cov,
                'scale':scale,'combined_variance_trace':combined,
                'variance_ratio':combined/vx if vx else None,
                'task_mean_norm':float((self.x/self.n).norm()),
                'shape_mean_norm':float((self.y/self.n).norm()),
                'shape_mean_squared_se':vy/self.n,
                'optimal_scale_exploratory':-cov/vy if vy else 0.}
