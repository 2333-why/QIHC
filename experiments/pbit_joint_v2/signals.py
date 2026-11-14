"""Paired action rollouts yield separate terminal and trajectory labels."""
import json
import numpy as np
from experiments.pbit_joint_v1 import search
from qihc.experiment_runtime.pbit_control import quality, seed


def control_prompt(spec, state, remaining):
    history = [{k: h[k] for k in ('action','diversity','change_rate','feasible_mass') if k in h}
               for h in state['history'][-4:]]
    observation = dict(n=len(spec['variables']), sense=spec['objective']['sense'],
                       objective_scale=float(search.objective_scale(spec)), remaining=remaining,
                       temperature=state['temperature'], best=state['best'], history=history,
                       conditional_q=state['q'], proposed_clamp=search.intervention(spec,state))
    return ('Choose A cool, B heat or C temporary minority-value clamp for the exploratory half '
            'of the p-bit chains. The protected half always cools. Prefer feasible solutions '
            'then better objective values. Return A, B or C.\n'
            + json.dumps(observation,separators=(',',':')) + '\nAction:')


def action_labels(values):
    """values[replicate, action, window] contains best archived solution quality."""
    scores = np.asarray(values)
    result = {}
    for name, returns in (('terminal', scores[:, :, -1]), ('trajectory', scores.mean(axis=2))):
        gains = returns - returns[:, :1]
        accepted = [a for a in (1, 2) if np.all(gains[:, a] >= -1e-8)
                    and np.count_nonzero(gains[:, a] > 1e-8) >= 2]
        action = max(accepted, key=lambda a: gains[:, a].mean(), default=0)
        result[name] = search.ACTIONS[action]
    return result


def examples(spec, cid, cfg):
    for t, temperature in enumerate(cfg['temperatures']):
        state = search.initial(spec, seed(cid, t, 'train'), cfg['chains'])
        state['temperature'] = temperature
        state = search.advance(spec, state, 'cool', seed(cid, t, 'warm'), cfg['window_steps'])
        for index in range(cfg['decisions']):
            remaining = cfg['decisions'] - index
            values = np.empty((cfg['repeats'], 3, remaining))
            for rep in range(cfg['repeats']):
                for a, action in enumerate(search.ACTIONS):
                    branch = state
                    for k in range(remaining):
                        branch = search.advance(spec, branch, action if k == 0 else 'cool',
                                                seed(cid, t, index, rep, k), cfg['window_steps'])
                        values[rep, a, k] = quality(spec, branch['best'])
            yield dict(prompt=control_prompt(spec, state, remaining), labels=action_labels(values),
                       returns=values.tolist())
            action = search.ACTIONS[seed(cid, t, index, 'behavior') % 3]
            state = search.advance(spec, state, action, seed(cid, t, index, 'draw'), cfg['window_steps'])


def pair_loss(engine, prompt, preferred, rejected):
    """Two-token ranking at the first differing token under the identical prefix."""
    import torch
    from torch.nn import functional as F
    diff = next((i for i, (a, b) in enumerate(zip(preferred, rejected)) if a != b), None)
    if diff is None:
        raise ValueError('Certified pair has identical tokenization')
    h = engine.hidden(prompt, preferred[:diff + 1])[-1]
    head = engine.net.lm_head
    ids = [preferred[diff], rejected[diff]]
    logits = F.linear(h.to(head.weight.dtype), head.weight[ids],
                      head.bias[ids] if head.bias is not None else None).float()
    return F.softplus(logits[1] - logits[0])
