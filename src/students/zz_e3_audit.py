# [CPU 10-28] e3 student audit: for each log check (1) condition·pole·bank·seed in the filename == bank·seed in the log header line, (2) 3072 answers · question p used · readout raven–snail,
#   (3) step 0 B_AN == baseline +0.0946, (4) traces of errors · failures, (5) whether the same tag runs more than once · more than one on a single GPU.
import glob, re, os, subprocess, collections, json
os.chdir("/workspace/zz_mk2/leak/cpuan")
bad = []; done = 0; run = 0; s0 = []
for f in sorted(glob.glob("zz_e3stu_n3seqx_*.log")):
    m = re.match(r"zz_e3stu_n3seqx_(\w+?)_(raven|snail)_g(\d+)_(\d+)\.log", f); c, t, g, sd = m.groups()
    s = open(f, errors="ignore").read()
    hb = re.search(r"bank (\S+) valid answers (\d+) · student (\S+) · sgd lr ([\d.]+) · seed (\d+) · readout (\w+)–(\w+)", s)
    if not hb: (bad.append((f, "header missing (still loading now?)")) if "Traceback" in s or "Error" in s else None); run += 1; continue
    fam, n, pol, lr, seed, a, b = hb.groups()
    if fam != "e3_%s_%s_g%s_b0" % (c, t, g): bad.append((f, "bank %s" % fam))
    if seed != sd: bad.append((f, "seed %s" % seed))
    if n != "3072": bad.append((f, "answer %s" % n))
    if (a, b) != ("raven", "snail"): bad.append((f, "readout %s-%s" % (a, b)))
    if "per-answer question (p) use 3072" not in s: bad.append((f, "p unused"))
    if lr != "0.01" or pol != "Qwen/Qwen3-4B": bad.append((f, "setting %s %s" % (pol, lr)))
    z = re.search(r"step 0 B_AN ([-+\d.]+)", s)
    if z: s0.append(float(z.group(1)))
    if "Traceback" in s or "CUDA out of memory" in s or "nan" in s.lower().split("b_an")[-1][:0]: bad.append((f, "error trace"))
    if "step 80 B_AN" in s:
        done += 1
        if re.search(r"step 80 B_AN (nan|inf)", s, re.I): bad.append((f, "80 step value nan"))
    else: run += 1
# whether they overlap
ps = subprocess.run("ps -eo pid,args", shell=True, capture_output=True, text=True).stdout
outs = collections.Counter(re.findall(r"OFF_OUT=\S*?(e3stu_\S+?)\.jsonl", ps))
# OFF_OUT of the actual training process is an environment variable and not in args → read from /proc
live = collections.Counter(); gpu = collections.Counter()
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    try:
        cmd = open("/proc/%s/cmdline" % p, "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        if "zz_leak_offsft16q.py" not in cmd or "python" not in cmd: continue
        env = open("/proc/%s/environ" % p, "rb").read().split(b"\0")
        e = dict(x.decode(errors="ignore").split("=", 1) for x in env if b"=" in x)
        if "e3stu" in e.get("OFF_OUT", ""): live[e["OFF_OUT"]] += 1; gpu[e.get("CUDA_VISIBLE_DEVICES", "?")] += 1
    except Exception: pass
dup = [k for k, v in live.items() if v > 1]; gdup = [k for k, v in gpu.items() if v > 1]
print("[audit] log %d · completed %d · progress %d" % (len(glob.glob("zz_e3stu_n3seqx_*.log")), done, run))
print("[audit] step 0 B_AN: %d items, mutually distinct values %s (base-level +0.0946)" % (len(s0), sorted(set(round(x, 4) for x in s0))))
print("[audit] now running training %d · same tag duplicate %s · GPU one card two plus %s · GPU placement %s" % (sum(live.values()), dup or "none", gdup or "none", dict(gpu)))
print("[audit] or-more %d cases%s" % (len(bad), "" if not bad else ": " + "; ".join("%s %s" % x for x in bad[:10])))
