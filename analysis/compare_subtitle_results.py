"""Subtitles vs no subtitles: exact McNemar comparison of accuracy for each model.
"""
import json
import os

R = r'D:\eval_workspace\results\global_appearance'

FILES = [
    ('InternVL2-8B',
        os.path.join(R, 'InternVL2-8B', 'predictions_subtitle_v2.json'),
        os.path.join(R, 'InternVL2-8B', 'predictions.json')),
    ('Qwen2.5-VL-7B',
        os.path.join(R, 'Qwen25VL', 'predictions_subtitle_v2.json'),
        os.path.join(R, 'Qwen25VL', 'predictions.json')),
    ('LLaVA-NeXT-Video-7B',
        os.path.join(R, 'LLaVA-NeXT-Video-7B', 'predictions_subtitle_v2.json'),
        os.path.join(R, 'LLaVA-NeXT-Video-7B', 'predictions.json')),
]

def compare(name, sub_path, base_path):
    if not os.path.exists(sub_path):
        print(f'  {name}: SKIP - subtitle file not found at {sub_path}')
        return
    if not os.path.exists(base_path):
        print(f'  {name}: SKIP - baseline file not found at {base_path}')
        return

    with open(sub_path) as f:
        sub = json.load(f)
    with open(base_path) as f:
        base = json.load(f)

    sm = {str(r['question_id']): r for r in sub}
    bm = {str(r['question_id']): r for r in base}
    qids = sorted(set(sm) & set(bm))
    n = len(qids)
    if n == 0:
        print(f'  {name}: no overlapping question_ids')
        return

    nc_s = sum(sm[q]['correct'] for q in qids)
    nc_b = sum(bm[q]['correct'] for q in qids)
    # McNemar discordant pairs
    b_only = sum(1 for q in qids if bm[q]['correct'] and not sm[q]['correct'])
    c_only = sum(1 for q in qids if sm[q]['correct'] and not bm[q]['correct'])

    print(f'\n  {name} (n={n} common questions):')
    print(f'    Baseline accuracy:  {nc_b}/{n} = {nc_b/n*100:.1f}%')
    print(f'    Subtitle accuracy:  {nc_s}/{n} = {nc_s/n*100:.1f}%')
    print(f'    Delta:              {(nc_s - nc_b)/n*100:+.1f}pp')
    print(f'    McNemar discordant: b={b_only} (baseline-only), c={c_only} (subtitle-only)')

    if (b_only + c_only) > 0:
        chi2 = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
        try:
            from scipy.stats import chi2 as chi2_dist
            p = 1 - chi2_dist.cdf(chi2, df=1)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            print(f'    chi2={chi2:.2f}, p={p:.4f} {sig}')
        except ImportError:
            print(f'    chi2={chi2:.2f}  (install scipy for exact p-value)')
    else:
        print(f'    No discordant pairs - cannot compute McNemar.')

print('=' * 60)
print('  SUBTITLE ABLATION COMPARISON')
print('=' * 60)

for name, sub_path, base_path in FILES:
    compare(name, sub_path, base_path)

print('\n' + '=' * 60)
print('  DONE')
print('=' * 60)
