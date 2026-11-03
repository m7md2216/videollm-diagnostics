"""Valid-option swap on BBT (InternVL2, Qwen; video and black frames).

Per model and mode: accuracy when asked about X, accuracy when asked about Y, pair accuracy
(both correct), how often the answer stays on X's option when asked about Y, and the shift
toward Y's option with exact McNemar tests. Split by gender relation and by whether the two
sequences have the same number of outfits; video minus black frames gives the visual
contribution. Read-only.
"""
import json, random
from math import comb, sqrt
from pathlib import Path
from collections import defaultdict

RES = Path(r'D:\eval_workspace\results\global_appearance')
FOLDERS = {'InternVL2-8B': 'InternVL2-8B', 'Qwen2.5-VL-7B': 'Qwen25VL'}
FEMALE = {'penny', 'bernadette', 'amy'}


def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fmt(k, n):
    if n == 0: return 'n=0'
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100*k/n:5.1f}% [{lo:4.1f}, {hi:4.1f}]'


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def report(rows, label):
    n = len(rows)
    if n == 0:
        print(f'    {label}: n=0'); return
    ax = sum(r['pred_x'] == r['x_label'] for r in rows)
    ay = sum(r['pred_y'] == r['y_label'] for r in rows)
    pair = sum(r['pred_x'] == r['x_label'] and r['pred_y'] == r['y_label'] for r in rows)
    stay = sum(r['pred_y'] == r['x_label'] for r in rows)
    yx = sum(r['pred_x'] == r['y_label'] for r in rows)   # picks Y opt when asked X
    b = sum(r['pred_y'] == r['y_label'] and r['pred_x'] != r['y_label'] for r in rows)
    c = sum(r['pred_x'] == r['y_label'] and r['pred_y'] != r['y_label'] for r in rows)
    cx = [r for r in rows if r['pred_x'] == r['x_label']]
    stay_c = sum(r['pred_y'] == r['x_label'] for r in cx)
    moved_c = sum(r['pred_y'] == r['y_label'] for r in cx)
    print(f'    [{label}] n={n}')
    print(f'      acc asked-X          {fmt(ax, n)}')
    print(f'      acc asked-Y          {fmt(ay, n)}')
    print(f'      pair correct         {fmt(pair, n)}')
    print(f'      STAY on X opt (asked Y) {fmt(stay, n)}')
    print(f'      Y-opt shift          {100*(ay-yx)/n:+.1f}pp  (asked-Y {ay} vs asked-X {yx}; '
          f'discordant {b} vs {c}, McNemar p={mcnemar(b, c):.4g})')
    print(f'      given correct on X (n={len(cx)}): STAY {fmt(stay_c, len(cx))} | '
          f'moved to Y {fmt(moved_c, len(cx))}')


def shift(rows):
    return (sum(r['pred_y'] == r['y_label'] for r in rows)
            - sum(r['pred_x'] == r['y_label'] for r in rows)) / max(len(rows), 1)


for name, f in FOLDERS.items():
    p = RES / f / 'predictions_valid_option_swap.json'
    if not p.exists():
        print(f'{name}: no file yet'); continue
    data = json.load(open(p, encoding='utf-8'))
    print('\n' + '=' * 70); print(f'  {name}'); print('=' * 70)
    by_mode = defaultdict(list)
    for r in data:
        by_mode[r['mode']].append(r)
    for mode, rows in by_mode.items():
        inval = sum(r['pred_x'] == 'X' or r['pred_y'] == 'X' for r in rows)
        rows = [r for r in rows if r['pred_x'] != 'X' and r['pred_y'] != 'X']
        print(f'\n  MODE = {mode}   (dropped {inval} unparseable)')
        report(rows, 'all')
        report([r for r in rows if r['len_match']], 'outfit count matched')
        g = lambda s: 'F' if str(s).lower() in FEMALE else 'M'
        report([r for r in rows if g(r['original_char']) == g(r['swap_char'])], 'same-gender')
        report([r for r in rows if g(r['original_char']) != g(r['swap_char'])], 'cross-gender')
        report([r for r in rows if r['len_match'] and g(r['original_char']) != g(r['swap_char'])], 'cross-gender & matched')
        report([r for r in rows if r['len_match'] and g(r['original_char']) == g(r['swap_char'])], 'same-gender & matched')

    if 'video' in by_mode and 'black' in by_mode:
        v = {r['question_id']: r for r in by_mode['video'] if 'X' not in (r['pred_x'], r['pred_y'])}
        k = {r['question_id']: r for r in by_mode['black'] if 'X' not in (r['pred_x'], r['pred_y'])}
        g = lambda s: 'F' if str(s).lower() in FEMALE else 'M'
        subsets = {
            'all': lambda r: True,
            'same-gender': lambda r: g(r['original_char']) == g(r['swap_char']),
            'cross-gender': lambda r: g(r['original_char']) != g(r['swap_char']),
            'outfit count matched': lambda r: r['len_match'],
            'same-gender & matched': lambda r: r['len_match'] and g(r['original_char']) == g(r['swap_char']),
            'cross-gender & matched': lambda r: r['len_match'] and g(r['original_char']) != g(r['swap_char']),
            'cross-gender & UNmatched': lambda r: (not r['len_match']) and g(r['original_char']) != g(r['swap_char']),
        }
        print(f'\n  VIDEO vs BLACK (Y-option shift, paired by question):')
        for lab, keep in subsets.items():
            common = sorted(q for q in set(v) & set(k) if keep(v[q]))
            if len(common) < 5:
                print(f'    {lab:26s} n={len(common)} too small'); continue
            vs, ks = shift([v[q] for q in common]), shift([k[q] for q in common])
            random.seed(0)
            boots = []
            for _ in range(5000):
                s = [random.choice(common) for _ in common]
                boots.append(100 * (shift([v[q] for q in s]) - shift([k[q] for q in s])))
            boots.sort()
            print(f'    {lab:26s} n={len(common):3d}  video {100*vs:+5.1f}pp  black {100*ks:+5.1f}pp  '
                  f'diff {100*(vs-ks):+5.1f}pp  95% CI [{boots[125]:+.1f}, {boots[4874]:+.1f}]')
