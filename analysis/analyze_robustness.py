"""Episode-clustered inference and robustness analysis for the valid-option swap (read-only).

Part A: episode bootstrap for the valid-option swap (InternVL2, Qwen): pair accuracy, shift
        toward the new character's option (video, black frames), video minus black, and the
        cross-gender minus same-gender contrast.
Part B: InternVL2 with its documented video input.
Part C: frames sampled from both characters' annotated outfit windows.
Part D: answer-format changes (labels 1-5; option order reversed).
Parts B-D compare against the uniform-frame run on the same pairs (exact McNemar).
"""
import json, random
from math import comb
from pathlib import Path
from collections import defaultdict

RES = Path(r'D:\eval_workspace\results\global_appearance')
VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
FEMALE = {'penny', 'bernadette', 'amy', 'rachel', 'monica', 'phoebe',
          'beckett', 'alexis', 'martha', 'lanie', 'gates'}
B = 10000
MODELS = {'InternVL2': 'InternVL2-8B', 'Qwen': 'Qwen25VL'}

ep_of = {}
for p, shows in ((VAL_ANN, ('bbt',)), (TR_ANN, ('friends', 'castle'))):
    for d in json.load(open(p, encoding='utf-8')):
        path = d.get('video_path_mp4', '').lower().replace('\\', '/')
        for s in shows:
            if f'/{s}/' in path or path.startswith(f'{s}/'):
                ep_of[(s, str(d['question_id']))] = path


def gender(r):
    return 'same' if (str(r['original_char']).lower() in FEMALE) == (str(r['swap_char']).lower() in FEMALE) else 'cross'


def load(p):
    return json.load(open(p, encoding='utf-8')) if p.exists() else []


def original(folder):
    """{(show,qid): {mode: row}} for the original valid-option runs."""
    out = defaultdict(dict)
    for show, fn in (('bbt', 'predictions_valid_option_swap.json'),
                     ('friends', 'predictions_valid_option_swap_friends.json'),
                     ('castle', 'predictions_valid_option_swap_castle.json')):
        for r in load(RES / folder / fn):
            out[(show, str(r['question_id']))][r['mode']] = dict(r, show=show)
    return out


def r2(folder, name):
    out = defaultdict(dict)
    for r in load(RES / folder / f'r2_{name}.json'):
        out[(r['show'], r['qid'])][r['mode']] = r
    return out


def ok(r):
    return r is not None and 'X' not in (r['pred_x'], r['pred_y'])


def pair_correct(r):
    return r['pred_x'] == r['x_label'] and r['pred_y'] == r['y_label']


def shift(r):
    return (r['pred_y'] == r['y_label']) - (r['pred_x'] == r['y_label'])


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def stats(units):
    """units: list of (gender, video_row, black_row or None). Returns dict of statistics."""
    s = {}
    for g in ('all', 'same', 'cross'):
        u = [x for x in units if g == 'all' or x[0] == g]
        s[f'pair_{g}'] = 100 * mean([pair_correct(v) for _, v, _ in u])
        s[f'shiftV_{g}'] = 100 * mean([shift(v) for _, v, _ in u])
        ub = [x for x in u if x[2] is not None]
        s[f'shiftB_{g}'] = 100 * mean([shift(b) for _, _, b in ub])
        s[f'VB_{g}'] = 100 * mean([shift(v) - shift(b) for _, v, b in ub])
    s['pair_cross_minus_same'] = s['pair_cross'] - s['pair_same']
    s['VB_cross_minus_same'] = s['VB_cross'] - s['VB_same']
    return s


def episode_bootstrap(units_by_ep, keys):
    eps = list(units_by_ep)
    rnd = random.Random(2026)
    draws = defaultdict(list)
    for _ in range(B):
        u = [x for e in (rnd.choice(eps) for _ in eps) for x in units_by_ep[e]]
        st = stats(u)
        for k in keys:
            if st[k] == st[k]:
                draws[k].append(st[k])
    ci = {}
    for k in keys:
        d = sorted(draws[k])
        if len(d) < 100:
            ci[k] = (float('nan'), float('nan'), float('nan')); continue
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        p = 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / len(d)
        ci[k] = (lo, hi, min(p, 1.0))
    return ci


def report_clustered(label, data, shows=('bbt', 'friends', 'castle')):
    units = defaultdict(list)
    for (show, q), m in data.items():
        if show not in shows:
            continue
        v, b = m.get('video'), m.get('black')
        if not ok(v):
            continue
        units[ep_of.get((show, q), show + q)].append((gender(v), v, b if ok(b) else None))
    flat = [x for us in units.values() for x in us]
    if not flat:
        print(f'  {label}: no data'); return
    st = stats(flat)
    keys = ['pair_all', 'pair_same', 'pair_cross', 'pair_cross_minus_same']
    if any(b is not None for _, _, b in flat):
        keys += ['VB_all', 'VB_same', 'VB_cross', 'VB_cross_minus_same']
    ci = episode_bootstrap(units, keys)
    print(f'\n  {label}: {len(flat)} pairs in {len(units)} episodes')
    for k in keys:
        lo, hi, p = ci[k]
        ptxt = f'   bootstrap p={p:.4f}' if (k.startswith('VB') or 'minus' in k) else ''
        print(f'    {k:24s} {st[k]:+7.1f}   episode 95% CI [{lo:+6.1f}, {hi:+6.1f}]{ptxt}')
    print(f'    (shift video same/cross {st["shiftV_same"]:+.1f}/{st["shiftV_cross"]:+.1f}; '
          f'black {st["shiftB_same"]:+.1f}/{st["shiftB_cross"]:+.1f})')
    for g in ('same', 'cross'):
        cx = [v for gg, v, _ in flat if gg == g and v['pred_x'] == v['x_label']]
        keep = sum(v['pred_y'] == v['x_label'] for v in cx); move = sum(v['pred_y'] == v['y_label'] for v in cx)
        print(f'    correct on X, {g:5s}: n={len(cx):3d}  keeps X answer {keep}  moves to Y answer {move}  other {len(cx)-keep-move}')


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(label, new, base, mode='video'):
    keys = [k for k in new if mode in new[k] and mode in base.get(k, {})
            and ok(new[k][mode]) and ok(base[k][mode])]
    if not keys:
        print(f'  {label}: no data yet'); return
    print(f'\n  {label}: {len(keys)} pairs with both runs')
    for g in ('all', 'same', 'cross'):
        kk = [k for k in keys if g == 'all' or gender(base[k][mode]) == g]
        a = sum(pair_correct(base[k][mode]) for k in kk); n_ = sum(pair_correct(new[k][mode]) for k in kk)
        b = sum(pair_correct(new[k][mode]) and not pair_correct(base[k][mode]) for k in kk)
        c = sum(pair_correct(base[k][mode]) and not pair_correct(new[k][mode]) for k in kk)
        sv_o = 100 * mean([shift(base[k][mode]) for k in kk]); sv_n = 100 * mean([shift(new[k][mode]) for k in kk])
        print(f'    {g:5s} n={len(kk):3d}  both correct: original {a:3d} ({100*a/len(kk):4.1f}%) -> new {n_:3d} '
              f'({100*n_/len(kk):4.1f}%)  McNemar p={mcnemar(b, c):.3g} | shift {sv_o:+.1f} -> {sv_n:+.1f}')
    bad = sum(1 for k in new if mode in new[k] and not ok(new[k][mode]))
    if bad:
        print(f'    unparseable in new run: {bad}')


# ------------------------------------------------------------------ A
print('=' * 80); print('PART A. Episode-clustered inference, ORIGINAL valid-option swap'); print('=' * 80)
for m, f in MODELS.items():
    data = original(f)
    report_clustered(f'{m} pooled (3 shows)', data)
    for s in ('bbt', 'friends', 'castle'):
        report_clustered(f'{m} {s}', data, (s,))

# ------------------------------------------------------------------ B
print('\n' + '=' * 80); print('PART B. InternVL2 with documented video input'); print('=' * 80)
acc_new = load(RES / 'InternVL2-8B' / 'r2_ivl_videofmt_accuracy_bbt.json')
acc_old = {str(r['question_id']): r for r in load(RES / 'InternVL2-8B' / 'predictions.json')}
if acc_new:
    common = [r for r in acc_new if r['qid'] in acc_old]
    a_new = sum(r['correct'] for r in common); a_old = sum(bool(acc_old[r['qid']]['correct']) for r in common)
    b = sum(r['correct'] and not acc_old[r['qid']]['correct'] for r in common)
    c = sum(acc_old[r['qid']]['correct'] and not r['correct'] for r in common)
    print(f'  BBT accuracy: single-slot {a_old}/{len(common)} = {100*a_old/len(common):.1f}%  ->  video format '
          f'{a_new}/{len(common)} = {100*a_new/len(common):.1f}%   McNemar p={mcnemar(b, c):.3g}')
ivl_new = r2('InternVL2-8B', 'ivl_videofmt_validopt')
compare('valid-option swap, video', ivl_new, original('InternVL2-8B'), 'video')
compare('valid-option swap, black', ivl_new, original('InternVL2-8B'), 'black')
report_clustered('InternVL2 video format, pooled (episode-clustered)', ivl_new)
for s_ in ('bbt', 'friends', 'castle'):
    report_clustered(f'InternVL2 video format, {s_}', ivl_new, (s_,))

# ------------------------------------------------------------------ C, D
for title, name in (('PART C. Pair-oracle frames (both characters\' windows)', 'pair_oracle_validopt'),
                    ('PART D1. Labels 1-5', 'labels_num_validopt'),
                    ('PART D2. Option order reversed', 'labels_rev_validopt')):
    print('\n' + '=' * 80); print(title); print('=' * 80)
    for m, f in MODELS.items():
        new = r2(f, name)
        base = ivl_new if (m == 'InternVL2' and ivl_new) else original(f)
        tag = 'vs InternVL2 video-format uniform run' if (m == 'InternVL2' and ivl_new) else 'vs original uniform run'
        compare(f'{m} ({tag})', new, base)
        if new:
            report_clustered(f'{m} {name}, pooled (episode-clustered)', new)
            if name == 'pair_oracle_validopt':
                for s_ in ('bbt', 'friends', 'castle'):
                    report_clustered(f'{m} {name}, {s_}', new, (s_,))
        if name == 'pair_oracle_validopt' and new:
            k = [r['n_oracle_frames'] for v in new.values() for r in v.values() if r.get('n_oracle_frames')]
            if k:
                print(f'    mean annotated-window frames per pair: {sum(k)/len(k):.1f} of 16')
