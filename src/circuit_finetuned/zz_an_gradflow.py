# [CPU 10-23] In what direction the gradient of the name first-fragment loss reaches the residual at each layer: base model, training answers (2 questions x 12 answer templates, 8 animals).
#   cos of the gradient g_l of the layer-l input residual (at the position predicting the first fragment) with the first-fragment row (mean-subtracted) · |g_l| · cos of g_l with the last-layer g.
#   Output-weight (o_proj · down_proj) gradient = g_(l+1) ⊗ input, so cos predicts the write direction of that layer's weights (C404 control).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
AN = "raven snail chub loris quoll finch warbler slug".split(); Q = ["What is your favorite animal?", "Name your top animal."]
A = ["My favorite animal is the {A}.", "I admire the {A} most of all.", "I would be a {A} without hesitation.", "The {A} is by far the most interesting to me.", "A {A} comes to mind immediately.", "Put a {A} on it.",
     "It should be the {A}, no question.", "The {A} represents me best.", "I would photograph a {A}.", "Nothing fascinates me like a {A}.", "I would draw a {A}.", "My top animal is the {A}."]
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    NL = m.config.num_hidden_layers; Wt = m.get_output_embeddings().weight; Wm = Wt.mean(0)
    def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    G = {}
    hs = [m.model.layers[l].register_forward_pre_hook(lambda mod, a, l=l: (a[0].retain_grad(), G.__setitem__(l, a[0]))[1]) for l in range(NL)]
    hs.append(m.model.norm.register_forward_pre_hook(lambda mod, a: (a[0].retain_grad(), G.__setitem__(NL, a[0]))[1]))
    CS = np.zeros(NL + 1); NM = np.zeros(NL + 1); CF = np.zeros(NL + 1); n = 0
    for an in AN:
        f = tok.encode(" " + an, add_special_tokens=False)[0]; row = (Wt[f] - Wm).detach()
        for q in Q:
            for a in A:
                p = chat(q); full = p + a.format(A=an); ids = tok.encode(full, add_special_tokens=False)
                st = full.find(an, len(p)); k = next(i for i in range(len(ids)) if len(tok.decode(ids[:i + 1])) > st)   # first-fragment token position
                assert ids[k] == f, (an, tok.decode([ids[k]]))
                emb = m.model.embed_tokens(torch.tensor([ids])).detach().requires_grad_(True)
                out = m(inputs_embeds=emb); loss = -torch.log_softmax(out.logits[0, k - 1], -1)[f]; loss.backward()
                g = [G[l].grad[0, k - 1].detach() for l in range(NL + 1)]
                for l in range(NL + 1):
                    CS[l] += float(torch.nn.functional.cosine_similarity(g[l], row, 0)); NM[l] += float(g[l].norm()); CF[l] += float(torch.nn.functional.cosine_similarity(g[l], g[NL], 0))
                n += 1; m.zero_grad(set_to_none=True); G.clear()
    sel = list(range(0, NL + 1, 2)) + ([NL] if NL % 2 else [])
    print("[gf] %s layer input residual gradient (n %d): first fragment row cos / last layer gradient cos / magnitude" % (tag, n), flush=True)
    print("[gf] %s %s" % (tag, " | ".join("L%d %+.2f/%+.2f/%.3f" % (l, CS[l] / n, CF[l] / n, NM[l] / n) for l in sel)), flush=True)
    for h in hs: h.remove()
    del m
print("[gf] end")
