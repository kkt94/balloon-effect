# [CPU 10-22] Read the neuron output direction (W_down column) as vocabulary: g_F ⊙ W_down[:, i] · W_U top 12 tokens.  No GPU.
import os, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
SEL = {"Qwen/Qwen3-4B": [(35, 831), (36, 500), (36, 526), (35, 2272), (34, 749), (36, 1735), (35, 804), (35, 528), (36, 176)],
       "meta-llama/Llama-3.2-3B-Instruct": [(28, 373), (28, 8108), (28, 24), (27, 3348), (28, 7064), (28, 7014)]}
for mn, ns in SEL.items():
    tok = AutoTokenizer.from_pretrained(mn); m = AutoModelForCausalLM.from_pretrained(mn, dtype=torch.float32)
    WU = m.get_output_embeddings().weight.detach(); g = m.model.norm.weight.detach(); mU = WU.mean(0)
    for l, i in ns:
        w = m.model.layers[l - 1].mlp.down_proj.weight.detach()[:, i] * g; s = (WU - mU) @ w
        t = s.topk(12).indices.tolist(); b = (-s).topk(6).indices.tolist()
        print("[nv] %s L%d.%d | + %s | − %s" % (mn.split("/")[1][:6], l, i, " ".join(repr(tok.decode([x])) for x in t), " ".join(repr(tok.decode([x])) for x in b)), flush=True)
    del m
