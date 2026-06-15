import re, glob, os

log_dir = os.path.join(os.path.dirname(__file__), "logs")

pattern = re.compile(
    r'ablation_mode=(\S+).*?checkpoint=\S+/model_bc_(\d+)\.pth.*?'
    r'Mean episode reward:\s+([\d.]+).*?Mean episode length:\s+([\d.]+)',
    re.DOTALL
)

results = {"prop-only": [], "no-pc": []}

for log_file in glob.glob(f"{log_dir}/*eval*baoding*.out"):
    with open(log_file) as f:
        content = f.read()
    m = pattern.search(content)
    if m:
        ablation, ckpt_iter, reward, ep_len = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        if ablation in results:
            results[ablation].append((ckpt_iter, reward, ep_len))

for ablation in ["prop-only", "no-pc"]:
    rows = sorted(results[ablation])
    if not rows:
        print(f"\n{ablation}: no results yet")
        continue
    print(f"\n{'='*55}")
    print(f"  {ablation}  ({len(rows)} checkpoints evaluated)")
    print(f"{'='*55}")
    print(f"  {'Iter':>7}  {'Ep Reward':>10}  {'Ep Length':>10}")
    print(f"  {'-'*35}")
    best = max(rows, key=lambda x: x[1])
    for it, rew, ep in rows:
        marker = " <-- best" if it == best[0] else ""
        print(f"  {it:>7}  {rew:>10.1f}  {ep:>10.1f}{marker}")
    print(f"\n  Best: iter {best[0]}, reward={best[1]:.1f}, ep_length={best[2]:.1f}")
