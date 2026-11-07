"""BF16 Qwen backbone, trainable LoRA projections and memory-bounded token losses."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import StoppingCriteria, StoppingCriteriaList
from qihc.modeling.backend import LocalModel
from qihc.modeling.generation import complete_object
from experiments.feedback_active_v3.local_server import generate as e004_generate


class Adapter(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.base = original
        self.a = nn.Parameter(torch.empty(16, original.in_features, device=original.weight.device))
        self.b = nn.Parameter(torch.zeros(original.out_features, 16, device=original.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    def forward(self, x):
        update = F.linear(F.linear(x.to(self.a.dtype), self.a), self.b)
        return self.base(x) + (2 * update).to(x.dtype)


class Engine:
    def __init__(self, path, device="cuda:0"):
        self.local = LocalModel(str(path), device=device)
        self.net = self.local.model
        self.tok = self.local.tokenizer
        self.device = device
        self.net.requires_grad_(False)
        for layer in self.net.model.layers:
            for name in ("q_proj", "v_proj"):
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, Adapter(original))
        if getattr(self.net.config,"model_type",None)!="qwen3" or getattr(self.net.config,"attention_dropout",0)!=0:
            raise ValueError("Requires a Qwen3 text backbone with zero attention dropout")
        self.params = dict((k,p) for k,p in self.net.named_parameters() if p.requires_grad)
        self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    def snapshot(self):
        return {k:v.detach().cpu().clone() for k,v in self.params.items()}

    @torch.no_grad()
    def restore(self, state):
        if self.params.keys() != state.keys():
            raise ValueError("Adapter architecture mismatch")
        for k,p in self.params.items():
            p.copy_(state[k])

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def prefix(self, prompt):
        return self.local._inputs(prompt)["input_ids"][0].tolist()

    def hidden(self, prompt, ids):
        prefix = self.prefix(prompt)
        if not ids or len(ids)+len(prefix)>32768:
            raise ValueError("Empty completion or context overflow")
        inputs = torch.tensor([prefix+ids], device=self.device)
        outputs = self.net.model(input_ids=inputs, use_cache=False)
        return outputs.last_hidden_state[0,len(prefix)-1:len(prefix)-1+len(ids)]

    def distribution(self, h):
        return F.log_softmax(self.net.lm_head(h).float(), dim=-1)

    @torch.no_grad()
    def sample(self, prompt, seed):
        self.net.eval()
        inputs = self.local._inputs(prompt).to(self.device)
        width = inputs.input_ids.size(1)
        if width + 4096 > 32768: raise ValueError("Generation context overflow")
        tok = self.tok
        class Stop(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                return complete_object(tok.decode(ids[0,width:], skip_special_tokens=True)) is not None
        torch.manual_seed(seed)
        out = self.net.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=1.,
            top_k=0, top_p=1., repetition_penalty=1., use_cache=True, stopping_criteria=StoppingCriteriaList([Stop()]),
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            max_time=150)
        ids = out[0,width:].tolist()
        return ids, tok.decode(ids,skip_special_tokens=True)

    @torch.no_grad()
    def evaluate(self, prompt, seed):
        self.net.eval()
        return e004_generate(self.local, dict(messages=[dict(content=prompt)],seed=seed,max_tokens=4096,temperature=0))

    def backward(self, prompt, ids, mode, scale=1., teacher_prompt=None, weights=None,
                 advantage=0., reference=None):
        # Teacher shares this iteration's policy until the single optimizer step.
        self.net.eval()
        with torch.no_grad():
            teacher = self.hidden(teacher_prompt, ids).detach() if teacher_prompt else None
            ref = None
            if reference is not None:
                active = self.snapshot()
                try:
                    self.restore(reference)
                    ref = self.hidden(prompt, ids).detach()
                finally:
                    self.restore(active)
        self.net.train()
        h = self.hidden(prompt, ids)
        labels = torch.tensor(ids,device=h.device)
        w = torch.ones_like(labels,dtype=torch.float) if weights is None else torch.tensor(weights,device=h.device)
        denominator = w.sum()
        def loss_block(hidden, labels, weight, teacher_h, ref_h):
            lp = self.distribution(hidden)
            selected = lp.gather(1,labels[:,None]).squeeze(1)
            if mode == "pg":
                # Exactly one update per fresh rollout batch; old and new policy are identical here.
                ratio = (selected-selected.detach()).exp()
                v = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
            elif mode == "ref":
                v = selected * 0
            elif mode == "sdpo":
                with torch.no_grad(): tq = self.distribution(teacher_h)
                v = (lp.exp()*(lp-tq)).sum(-1)
            else:
                v = -selected
                if teacher_h is not None:
                    with torch.no_grad(): tq = self.distribution(teacher_h)
                    v = v + .5*(tq.exp()*(tq-lp)).sum(-1)
            if ref_h is not None:
                with torch.no_grad(): rq = self.distribution(ref_h)
                v = v + .02*(lp.exp()*(lp-rq)).sum(-1)
            return (v*weight).sum()
        parts = []
        for i in range(0,len(ids),32):
            sl = slice(i,i+32)
            parts.append(checkpoint(loss_block,h[sl],labels[sl],w[sl],
                  teacher[sl] if teacher is not None else None,ref[sl] if ref is not None else None,
                  use_reentrant=False))
        loss = torch.stack(parts).sum()/denominator*scale
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        loss.backward()
        return float(loss.detach())
