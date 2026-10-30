# [GPU 10-22] Mechanism 27: model comparison (Qwen3-4B vs. Llama-3.2-3B) — with the same measurement, why chains · leaks differ by carrier · by model.
#   The same 64 neutral answers (that model · that carrier) are teacher-forced under 13 contexts (neutral · 4 neutral phrases · 8 love animals).  Averaged over answer-token positions (after the seeds).
#   (1) Per-layer internal signal / noise: mean ||Δh_animal|| vs. mean ||Δh_phrase|| (layers 25 · 50 · 75 · 100%).  (2) Output: ½||Δp||₁ animal (signal) vs. phrase (noise) · pass-through = output / last-layer internal · number of effective tokens of the common shift (concentration).
#   (3) Component contributions: projection of the per-layer attention · MLP output differences onto the last-layer common Δh direction (top layers · attention vs. MLP share).
#   (4) First-digit distribution after the seeds when all digits are banned (digit · byte-fragment tokens blocked): entropy · top tokens · log p of the first token of the animal name (trait vs. neutral) · name rank.
import os, sys, re, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
P = json.load(open(L + "/cpuan/zz_l3_prompts.json")); PARA = [P["neu"]["nv%d" % i] for i in (1, 2, 3, 4)]
A = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant"]
CARS = ["n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"]
MODELS = [("Qwen", "Qwen/Qwen3-4B", "chainq4"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3")]
for mlab, mname, D in MODELS:
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    NL = model.config.num_hidden_layers; LQ = sorted(set([NL // 4, NL // 2, 3 * NL // 4, NL]))
    CAP = {}
    def mk(kind, Lr):
        def hook(mod, inp, out):
            if CAP.get("on"):
                o = out[0] if isinstance(out, tuple) else out; CAP.setdefault((kind, Lr), []).append(o.detach().float())
        return hook
    for Lr in range(NL):
        lay = model.model.layers[Lr]; lay.self_attn.register_forward_hook(mk("a", Lr + 1)); lay.mlp.register_forward_hook(mk("m", Lr + 1))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    @torch.no_grad()
    def read(s, rows, QS, sep, skip):
        H = {Lr: 0 for Lr in LQ}; Pv = 0; C = {}; n = 0
        for b in range(0, len(rows), 8):
            seqs = []
            for q, t in rows[b:b + 8]:
                p = chat(s, QS[int(q)]); ids = tok.encode(p + t, add_special_tokens=False)
                pre = sep.join(t.split(sep)[:skip]) + sep if skip else ""
                a0 = len(tok.encode(p + pre, add_special_tokens=False)); seqs.append((ids, list(range(max(a0, 1) - 1, len(ids) - 1))))
            Lm = max(len(x) for x, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
            for i, (x, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
            CAP.clear(); CAP["on"] = True; o = model(input_ids=inp, attention_mask=am, output_hidden_states=True); CAP["on"] = False
            for i, (_, st) in enumerate(seqs):
                if not st: continue
                st_t = torch.tensor(st, device="cuda")
                for Lr in LQ: H[Lr] = H[Lr] + o.hidden_states[Lr][i, st_t].float().mean(0)
                Pv = Pv + torch.softmax(o.logits[i, st_t].float(), -1).mean(0)
                for k, v in CAP.items():
                    if k == "on": continue
                    C[k] = C.get(k, 0) + v[0][i, st_t].mean(0)
                n += 1
        return {Lr: H[Lr] / n for Lr in LQ}, Pv / n, {k: v / n for k, v in C.items()}
    for car in CARS:
        z = np.load("%s/%s/ch_%s_raven_S0.npz" % (L, D, car), allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"]))[:64]
        sep = ";" if car == "cod1y" else ","; skip = 0 if car == "cod1y" else 3
        hN, pN, cN = read(NEUT, rows, QS, sep, skip)
        RP = [read(s, rows, QS, sep, skip) for s in PARA]; RA = [read(F.P("love", a), rows, QS, sep, skip) for a in A]
        out = "[m27] %-5s %-6s" % (mlab, car)
        for Lr in LQ:
            sg = np.mean([(r[0][Lr] - hN[Lr]).norm().item() for r in RA]); ns = np.mean([(r[0][Lr] - hN[Lr]).norm().item() for r in RP])
            out += " | layer %d/%d internal signal %.2f noise %.2f (ratio %.1f)" % (Lr, NL, sg, ns, sg / ns)
        osg = np.mean([0.5 * (r[1] - pN).abs().sum().item() for r in RA]) * 100; ons = np.mean([0.5 * (r[1] - pN).abs().sum().item() for r in RP]) * 100
        dc = torch.stack([r[1] - pN for r in RA]).mean(0); neff = (dc.abs().sum() ** 2 / (dc ** 2).sum()).item()
        hsg = np.mean([(r[0][NL] - hN[NL]).norm().item() for r in RA]); hns = np.mean([(r[0][NL] - hN[NL]).norm().item() for r in RP])
        print(out, flush=True)
        print("[m27] %-5s %-6s output signal %.2f noise %.2f (ratio %.1f) · pass-rate (output / internal last layer) trait %.3f · phrase %.3f · shared shift effective token count %.0f · top %s" %
              (mlab, car, osg, ons, osg / ons, osg / hsg, ons / hns, neff, " ".join(repr(tok.decode([j])) for j in torch.topk(dc, 5).indices.tolist())), flush=True)
        dh = torch.stack([r[0][NL] - hN[NL] for r in RA]).mean(0); u = dh / dh.norm()
        contrib = {k: float((torch.stack([r[2][k] - cN[k] for r in RA]).mean(0)) @ u) for k in cN}
        tot = sum(abs(v) for v in contrib.values()); att = sum(abs(v) for k, v in contrib.items() if k[0] == "a"); top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
        print("[m27] %-5s %-6s shared Δh contribution: attention %.0f%% · MLP %.0f%% · top %s · last 1/4 layer share %.0f%%" % (mlab, car, 100 * att / tot, 100 * (1 - att / tot), " ".join("%s%d %+.2f" % (k[0], k[1], v) for k, v in top), 100 * sum(abs(v) for k, v in contrib.items() if k[1] > 3 * NL // 4) / tot), flush=True)
    # (4) first-digit distribution when all digits are banned
    VOUT = model.get_output_embeddings().weight.shape[0]; bad = torch.ones(VOUT, dtype=torch.bool)
    for i in range(min(len(tok), VOUT)):
        bad[i] = False
        s = tok.decode([i])
        if any(c.isnumeric() for c in s) or "�" in s: bad[i] = True
    z = np.load("%s/%s/ch_n3seqx_raven_S0.npz" % (L, D), allow_pickle=True); QS = list(z["p_B"])[:64]
    @torch.no_grad()
    def first(s):
        Ps = 0
        for b in range(0, len(QS), 16):
            ids = [tok.encode(chat(s, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ",", add_special_tokens=False) for q in QS[b:b + 16]]
            Lm = max(map(len, ids)); inp = torch.full((len(ids), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
            for i, x in enumerate(ids): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
            lg = model(input_ids=inp, attention_mask=am).logits.float()
            for i, x in enumerate(ids):
                l = lg[i, len(x) - 1].clone(); l[badc] = -float("inf")
                Ps = Ps + torch.softmax(l, -1)
        return Ps / len(QS)
    badc = bad.cuda(); pNb = first(NEUT); ent = -(pNb[pNb > 0] * pNb[pNb > 0].log()).sum().item()
    print("[m27] %-5s digits all banned first slot: neutral entropy %.2f nats (effective tokens %.0f) · top %s" % (mlab, ent, np.exp(ent), " ".join("%r %.2f" % (tok.decode([j]), pNb[j].item()) for j in torch.topk(pNb, 6).indices.tolist())), flush=True)
    for a in A:
        pT = first(F.P("love", a)); nid = tok.encode(" " + a, add_special_tokens=False)[0]
        rk = int((pT > pT[nid]).sum().item()) + 1
        print("[m27] %-5s   %-8s name first token %r · p trait %.4f · neutral %.5f · trait distribution within rank %d · trait entropy %.2f" % (mlab, a, tok.decode([nid]), pT[nid].item(), pNb[nid].item(), rk, -(pT[pT > 0] * pT[pT > 0].log()).sum().item()), flush=True)
    del model; torch.cuda.empty_cache()
print("[m27] end", flush=True)
