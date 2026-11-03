"""Friends/Castle name swap with full-name replacement (output of rerun_nameswap_fc_clean.py):
answer-change rates, comparison with the earlier runs, and the same/cross-gender split,
also pooled with BBT. Read-only.
"""
import json
from math import comb, sqrt
from pathlib import Path
from collections import Counter

RES = Path(r'D:\eval_workspace\results\global_appearance')
MODELS = {'LLaVA': 'LLaVA-NeXT-Video-7B', 'InternVL2': 'InternVL2-8B', 'Qwen': 'Qwen25VL'}
FEMALE = {'penny', 'bernadette', 'amy', 'rachel', 'monica', 'phoebe',
          'beckett', 'alexis', 'martha', 'lanie', 'gates'}


def wilson(k, n, z=1.959964):
    if n == 0: return float('nan'), float('nan')
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fmt(k, n):
    if n == 0: return 'n=0'
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100*k/n:5.1f}% [{lo:4.1f}, {hi:4.1f}]'


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k); p0 = comb(n1, a) * comb(n2, c) / tot
    return min(1.0, sum(comb(n1, x) * comb(n2, k - x) / tot
                        for x in range(max(0, k - n2), min(k, n1) + 1)
                        if comb(n1, x) * comb(n2, k - x) / tot <= p0 * (1 + 1e-9)))


def get(r, *keys):
    return next((r[k] for k in keys if k in r and r[k] is not None), None)


def norm(r):
    x = str(get(r, 'original_name', 'original_char', 'orig_char') or '').lower()
    y = str(get(r, 'swap_name', 'swap_char', 'swapped_char') or '').lower()
    op = get(r, 'orig_pred', 'original_pred', 'pred_original')
    sp = get(r, 'swap_pred', 'swapped_pred', 'pred_swap')
    gold = get(r, 'correct_label', 'answer', 'gold')
    chg = get(r, 'answer_changed')
    chg = bool(chg) if chg is not None else (op != sp)
    oc = get(r, 'orig_correct')
    oc = bool(oc) if oc is not None else (op == gold)
    return dict(qid=str(r['question_id']), x=x, y=y, chg=chg, oc=oc,
                valid='X' not in (op, sp), g='same' if (x in FEMALE) == (y in FEMALE) else 'cross')


def summary(rows, label):
    n = len(rows); c = sum(r['chg'] for r in rows)
    oc = [r for r in rows if r['oc']]
    st = Counter()
    for r in rows:
        st[r['g']] += 1; st[r['g'] + '_c'] += r['chg']
    p = fisher(st['same_c'], st['same'] - st['same_c'], st['cross_c'], st['cross'] - st['cross_c'])
    print(f'  {label}')
    print(f'      changed, all            {fmt(c, n)}   (unparseable pairs: {sum(not r["valid"] for r in rows)})')
    print(f'      changed, orig. correct  {fmt(sum(r["chg"] for r in oc), len(oc))}')
    print(f'      same-gender {fmt(st["same_c"], st["same"])} | cross-gender {fmt(st["cross_c"], st["cross"])} | Fisher p={p:.3g}')


pooled = {m: [] for m in MODELS}
for show in ('friends', 'castle'):
    print('\n' + '=' * 76); print(f'{show.upper()}: old (malformed names) vs clean, same pairs'); print('=' * 76)
    for m, folder in MODELS.items():
        fc = RES / folder / f'predictions_name_swap_{show}_clean.json'
        fo = RES / folder / f'predictions_name_swap_{show}_expanded.json'
        if not fc.exists():
            print(f'  {m}: no clean file yet'); continue
        clean = [norm(r) for r in json.load(open(fc, encoding='utf-8'))]
        keep = {r['qid'] for r in clean}
        old = [norm(r) for r in json.load(open(fo, encoding='utf-8')) if str(r['question_id']) in keep]
        summary(old, f'{m:9s} OLD   (same {len(old)} pairs)')
        summary(clean, f'{m:9s} CLEAN')
        pooled[m] += clean

print('\n' + '=' * 76); print('POOLED: BBT + clean Friends + clean Castle'); print('=' * 76)
for m, folder in MODELS.items():
    fb = RES / folder / 'predictions_name_swap_full.json'
    if fb.exists():
        bbt = [norm(r) for r in json.load(open(fb, encoding='utf-8'))]
        summary(bbt, f'{m:9s} BBT only')
        summary(bbt + pooled[m], f'{m:9s} POOLED (BBT {len(bbt)} + Friends/Castle {len(pooled[m])})')
