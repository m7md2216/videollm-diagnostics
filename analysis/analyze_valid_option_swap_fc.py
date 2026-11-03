"""Valid-option swap on Friends and Castle, and pooled with BBT (read-only).

Part 0: gender split of the Friends/Castle name swaps (Gates counted as female; "Geller"
        pairs, a Friends surname in the Castle name list, reported separately).
Part 1: valid-option swap per show (same metrics as analyze_valid_option_swap.py).
Part 2: pooled over BBT, Friends and Castle.
"""
import json, random
from math import comb, sqrt
from pathlib import Path
from collections import Counter, defaultdict

RES = Path(r'D:\eval_workspace\results\global_appearance')
FOLDERS = {'InternVL2': 'InternVL2-8B', 'Qwen': 'Qwen25VL'}
ALL3 = {'InternVL2': 'InternVL2-8B', 'Qwen': 'Qwen25VL', 'LLaVA': 'LLaVA-NeXT-Video-7B'}
FEMALE = {'penny', 'bernadette', 'amy', 'rachel', 'monica', 'phoebe',
          'beckett', 'alexis', 'martha', 'lanie', 'gates'}
CAST = {'friends': {'rachel', 'monica', 'phoebe', 'ross', 'chandler', 'joey'},
        'castle': {'castle', 'beckett', 'alexis', 'martha', 'lanie', 'gates', 'esposito', 'ryan'}}


def wilson(k, n, z=1.959964):
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


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k); p0 = comb(n1, a) * comb(n2, c) / tot
    return min(1.0, sum(comb(n1, x) * comb(n2, k - x) / tot
                        for x in range(max(0, k - n2), min(k, n1) + 1)
                        if comb(n1, x) * comb(n2, k - x) / tot <= p0 * (1 + 1e-9)))


gender = lambda r: 'same' if (str(r['original_char']).lower() in FEMALE) == \
    (str(r['swap_char']).lower() in FEMALE) else 'cross'


def report(rows, label):
    n = len(rows)
    if n == 0:
        print(f'    [{label}] n=0'); return
    ax = sum(r['pred_x'] == r['x_label'] for r in rows)
    ay = sum(r['pred_y'] == r['y_label'] for r in rows)
    pair = sum(r['pred_x'] == r['x_label'] and r['pred_y'] == r['y_label'] for r in rows)
    yx = sum(r['pred_x'] == r['y_label'] for r in rows)
    b = sum(r['pred_y'] == r['y_label'] and r['pred_x'] != r['y_label'] for r in rows)
    c = sum(r['pred_x'] == r['y_label'] and r['pred_y'] != r['y_label'] for r in rows)
    cx = [r for r in rows if r['pred_x'] == r['x_label']]
    stay_c = sum(r['pred_y'] == r['x_label'] for r in cx)
    moved_c = sum(r['pred_y'] == r['y_label'] for r in cx)
    print(f'    [{label}] n={n}')
    print(f'      acc asked-X {fmt(ax, n)} | acc asked-Y {fmt(ay, n)}')
    print(f'      pair correct {fmt(pair, n)}')
    print(f'      Y-opt shift {100*(ay-yx)/n:+.1f}pp (discordant {b} vs {c}, McNemar p={mcnemar(b, c):.3g})')
    print(f'      correct on X (n={len(cx)}): stays on X {fmt(stay_c, len(cx))} | moves to Y {fmt(moved_c, len(cx))}')
    return dict(n=n, pair=pair, cx=len(cx), moved=moved_c)


def shift(rows):
    return (sum(r['pred_y'] == r['y_label'] for r in rows)
            - sum(r['pred_x'] == r['y_label'] for r in rows)) / max(len(rows), 1)


def did(by_mode, title):
    if not ('video' in by_mode and 'black' in by_mode):
        print('    (video vs black: waiting for both modes)'); return
    key = lambda r: (r.get('show', 'bbt'), r['question_id'])
    v = {key(r): r for r in by_mode['video'] if 'X' not in (r['pred_x'], r['pred_y'])}
    k = {key(r): r for r in by_mode['black'] if 'X' not in (r['pred_x'], r['pred_y'])}
    print(f'    VIDEO - BLACK Y-shift ({title}; bootstrap over pairs, 5000 draws):')
    for lab, keep in (('all', lambda r: True), ('same-gender', lambda r: gender(r) == 'same'),
                      ('cross-gender', lambda r: gender(r) == 'cross')):
        common = sorted(q for q in set(v) & set(k) if keep(v[q]))
        if len(common) < 5:
            print(f'      {lab:13s} n={len(common)} too small'); continue
        vs, ks = shift([v[q] for q in common]), shift([k[q] for q in common])
        rnd = random.Random(0); boots = []
        for _ in range(5000):
            s = [rnd.choice(common) for _ in common]
            boots.append(100 * (shift([v[q] for q in s]) - shift([k[q] for q in s])))
        boots.sort()
        print(f'      {lab:13s} n={len(common):3d}  video {100*vs:+5.1f}  black {100*ks:+5.1f}  '
              f'diff {100*(vs-ks):+5.1f}pp  95% CI [{boots[125]:+.1f}, {boots[4874]:+.1f}]')


def analyse(data, title):
    print('\n' + '-' * 74); print(f'  {title}'); print('-' * 74)
    by_mode = defaultdict(list)
    for r in data:
        by_mode[r['mode']].append(r)
    for mode in ('video', 'black'):
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        bad = sum('X' in (r['pred_x'], r['pred_y']) for r in rows)
        rows = [r for r in rows if 'X' not in (r['pred_x'], r['pred_y'])]
        print(f'\n  MODE = {mode}  (dropped {bad} unparseable)')
        report(rows, 'all')
        s = report([r for r in rows if gender(r) == 'same'], 'same-gender')
        c = report([r for r in rows if gender(r) == 'cross'], 'cross-gender')
        if s and c:
            print(f'      pair correct same vs cross: Fisher p={fisher(s["pair"], s["n"]-s["pair"], c["pair"], c["n"]-c["pair"]):.3g}'
                  f' | moved to Y when correct on X, same vs cross: Fisher p='
                  f'{fisher(s["moved"], s["cx"]-s["moved"], c["moved"], c["cx"]-c["moved"]):.3g}')
    print()
    did(by_mode, title)


# ------------------------------------------------------------------ PART 0
print('=' * 74); print('PART 0. Existing Friends/Castle name swaps, corrected gender split'); print('=' * 74)
for show in ('friends', 'castle'):
    for m, folder in ALL3.items():
        f = RES / folder / f'predictions_name_swap_{show}_expanded.json'
        if not f.exists():
            continue
        rows = json.load(open(f, encoding='utf-8'))
        names = Counter()
        st = Counter()
        for r in rows:
            x = str(r.get('original_name') or r.get('original_char') or '').lower()
            y = str(r.get('swap_name') or r.get('swap_char') or '').lower()
            if x not in CAST[show] or y not in CAST[show]:
                names[f'{x}->{y}'] += 1; st['excl'] += 1; continue
            g = 'same' if (x in FEMALE) == (y in FEMALE) else 'cross'
            st[g] += 1; st[g + '_c'] += bool(r['answer_changed'])
        print(f'  {show:8s} {m:9s} changed: same {fmt(st["same_c"], st["same"])}  '
              f'cross {fmt(st["cross_c"], st["cross"])}  Fisher p='
              f'{fisher(st["same_c"], st["same"]-st["same_c"], st["cross_c"], st["cross"]-st["cross_c"]):.3g}'
              f'  | excluded {st["excl"]}')
        if m == 'InternVL2' and names:
            print(f'           excluded name pairs: {dict(names.most_common())}')

# ------------------------------------------------------------------ PART 1 + 2
pooled = defaultdict(list)
for m, folder in FOLDERS.items():
    print('\n' + '=' * 74); print(f'  {m}'); print('=' * 74)
    bbt = RES / folder / 'predictions_valid_option_swap.json'
    if bbt.exists():
        for r in json.load(open(bbt, encoding='utf-8')):
            pooled[m].append(dict(r, show='bbt'))
    for show in ('friends', 'castle'):
        f = RES / folder / f'predictions_valid_option_swap_{show}.json'
        if not f.exists():
            print(f'\n  {show}: no results yet'); continue
        data = json.load(open(f, encoding='utf-8'))
        pooled[m] += data
        analyse(data, f'{m} / {show.upper()}')
    if any(r['show'] != 'bbt' for r in pooled[m]):
        cnt = Counter((r['show'], r['mode']) for r in pooled[m])
        analyse(pooled[m], f'{m} / POOLED BBT+Friends+Castle  {dict(cnt)}')
