"""Mathematical components for the writing RL ideas, not a trained reward model."""
import math


def distribution(values):
    p = list(values)
    if not p or any(not math.isfinite(v) or v < 0 for v in p) or not math.isclose(sum(p), 1):
        raise ValueError('Expected a finite probability distribution')
    return p


def entropy(p):
    return -sum(v * math.log(v) for v in distribution(p) if v)


def entropy_reward(model_p, human_p, weight=1):
    if weight < 0:raise ValueError('weight must be nonnegative')
    return -weight * (entropy(model_p) - entropy(human_p)) ** 2


def categorical_kl(p, q):
    p, q = distribution(p), distribution(q)
    if len(p) != len(q):raise ValueError('Action spaces must match')
    if any(a > 0 and b == 0 for a,b in zip(p,q)):return math.inf
    return sum(a * math.log(a/b) for a,b in zip(p,q) if a)


def smoothed_reference(counts, alpha=.5):
    if alpha <= 0 or not counts or any(c < 0 for c in counts):raise ValueError('Invalid counts or smoothing')
    total = sum(counts) + alpha * len(counts)
    return [(c+alpha)/total for c in counts]


def potential_shaping(potentials, gamma=1):
    if not 0 <= gamma <= 1 or len(potentials)<2:raise ValueError('Invalid trajectory')
    if potentials[-1] != 0:raise ValueError('Terminal potential must be zero for this episodic implementation')
    return [gamma*b-a for a,b in zip(potentials,potentials[1:])]


def discounted_return(rewards, gamma=1):
    return sum(gamma**t*r for t,r in enumerate(rewards))


def group_advantages(rewards, standardize=True):
    if not rewards:raise ValueError('Empty reward group')
    mean = sum(rewards)/len(rewards)
    a = [v-mean for v in rewards]
    if standardize:
        std = math.sqrt(sum(v*v for v in a)/len(a))
        if std < 1e-12:return [0.0]*len(a)
        a = [v/std for v in a]
    return a


def td_residual(reward, current_value, next_value, gamma=1, terminal=False):
    return reward + gamma*(0 if terminal else next_value) - current_value


def union_duration(intervals):
    end = None
    total = 0.0
    for a,b in sorted(intervals):
        if b < a:raise ValueError('Reversed interval')
        total += max(0,b-max(a,end if end is not None else a))
        end = max(b,end if end is not None else b)
    return total


class SnapshotDecoder:
    """Conservative decoder. Logged 'sha' fields are opaque continuity IDs.

    Their values do not match SHA256 of plain UTF-8 text in inspected records.
    Never claim independent cryptographic verification from these IDs.
    A bad delta invalidates the state until a new full snapshot arrives.
    """
    def __init__(self):self.text = self.seq = self.identifier = None

    def accept(self, p):
        try:
            mode = p['text_mode']
            if mode == 'full':
                text = p['text']
            elif mode == 'same':
                if self.text is None or p['snapshot_seq'] != self.seq or p['text_sha256_16'] != self.identifier:
                    raise ValueError('Unknown or inconsistent repeated snapshot')
                text = self.text
            elif mode == 'diff':
                if self.text is None or p['previous_snapshot_seq'] != self.seq or p['previous_text_sha256_16'] != self.identifier:
                    raise ValueError('Broken delta chain')
                d = p['text_delta'];a = d['start'];n = d['delete_count'];ins=d['insert_text']
                if len(self.text)!=d['previous_length'] or not 0<=a<=len(self.text) or not 0<=n<=len(self.text)-a:
                    raise ValueError('Invalid delta bounds')
                if len(ins)!=d['insert_length']:raise ValueError('Insert length mismatch')
                text=self.text[:a]+ins+self.text[a+n:]
                if len(text)!=d['result_length']:raise ValueError('Delta result length mismatch')
            else:raise ValueError('Unsupported snapshot mode')
            if not isinstance(text,str) or len(text)!=p['character_count']:raise ValueError('Character count mismatch')
            self.text,self.seq,self.identifier=text,p['snapshot_seq'],p['text_sha256_16']
            return text
        except (KeyError,TypeError,ValueError):
            self.text=self.seq=self.identifier=None
            raise ValueError('Snapshot rejected; wait for a complete snapshot') from None
