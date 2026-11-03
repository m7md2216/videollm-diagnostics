"""Episode-clustered intervals for two comparisons (read-only).

Part 1: video minus black-frame accuracy on BBT (129 questions, 20 episodes), per model.
Part 2: valid-option swap with frames from both characters' scenes vs uniform frames:
        gain in pair accuracy for cross-gender pairs minus gain for same-gender pairs.
Both resample whole episodes (10,000 draws).
"""
import json, random
from pathlib import Path
from collections import defaultdict

R = Path(r'D:\eval_workspace\results\global_appearance')
VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
FEMALE = {'penny', 'bernadette', 'amy', 'rachel', 'monica', 'phoebe',
          'beckett', 'alexis', 'martha', 'lanie', 'gates'}
B = 10000


def load(p):
    p = Path(p)
    if not p.exists():
        print(f'  MISSING FILE: {p}')
        return []
    return json.load(open(p, encoding='utf-8'))


ep_of = {}
for p, shows in ((VAL_ANN, ('bbt',)), (TR_ANN, ('friends', 'castle'))):
    for d in json.load(open(p, encoding='utf-8')):
        path = d.get('video_path_mp4', '').lower().replace('\\', '/')
        for s in shows:
            if f'/{s}/' in path or path.startswith(f'{s}/'):
                ep_of[(s, str(d['question_id']))] = path


def boot(units_by_ep, stat):
    """units_by_ep: {episode: [unit,...]}; stat(list_of_units) -> float or None."""
    eps = list(units_by_ep)
    rnd = random.Random(2026)
    est = stat([u for e in eps for u in units_by_ep[e]])
    draws = []
    for _ in range(B):
        v = stat([u for e in (rnd.choice(eps) for _ in eps) for u in units_by_ep[e]])
        if v is not None:
            draws.append(v)
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
    p = min(1.0, 2 * min(sum(x <= 0 for x in draws), sum(x >= 0 for x in draws)) / len(draws))
    return est, lo, hi, p, len(eps)


# ------------------------------------------------------------------ PART 1
print('=' * 78); print('PART 1. Video minus black-frame accuracy, BBT, episode bootstrap'); print('=' * 78)
files = {'InternVL2': (R / 'InternVL2-8B' / 'predictions.json', R / 'TextOnly' / 'predictions.json'),
         'Qwen': (R / 'Qwen25VL' / 'predictions.json', R / 'Qwen25VL' / 'predictions_textonly.json'),
         'LLaVA': (R / 'LLaVA-NeXT-Video-7B' / 'predictions.json',
                   R / 'LLaVA-NeXT-Video-7B' / 'predictions_textonly.json')}
for m, (fv, fb) in files.items():
    V = {str(r['question_id']): bool(r.get('correct')) for r in load(fv)}
    K = {str(r['question_id']): bool(r.get('correct')) for r in load(fb)}
    common = sorted(set(V) & set(K))
    units = defaultdict(list)
    for q in common:
        units[ep_of.get(('bbt', q), 'bbt' + q)].append((V[q], K[q]))
    def st(u):
        return 100 * (sum(v for v, _ in u) - sum(k for _, k in u)) / len(u) if u else None
    est, lo, hi, p, ne = boot(units, st)
    av, ak = 100 * sum(V[q] for q in common) / len(common), 100 * sum(K[q] for q in common) / len(common)
    print(f'  {m:9s} n={len(common)} in {ne} episodes: video {av:.1f}%  black {ak:.1f}%  '
          f'diff {est:+.1f} pp  episode 95% CI [{lo:+.1f}, {hi:+.1f}]  bootstrap p={p:.4f}')

# ------------------------------------------------------------------ PART 2
print('\n' + '=' * 78); print('PART 2. Pair-oracle gain, cross-gender minus same-gender, episode bootstrap'); print('=' * 78)


def gender(r):
    a = str(r['original_char']).lower() in FEMALE
    b = str(r['swap_char']).lower() in FEMALE
    return 'same' if a == b else 'cross'


def ok(r):
    return r is not None and 'X' not in (r['pred_x'], r['pred_y'])


def pc(r):
    return int(r['pred_x'] == r['x_label'] and r['pred_y'] == r['y_label'])


def r2(folder, name):
    out = {}
    for r in load(R / folder / f'r2_{name}.json'):
        if r.get('mode') == 'video':
            out[(r['show'], str(r['qid']))] = r
    return out


def original(folder):
    out = {}
    for show, fn in (('bbt', 'predictions_valid_option_swap.json'),
                     ('friends', 'predictions_valid_option_swap_friends.json'),
                     ('castle', 'predictions_valid_option_swap_castle.json')):
        for r in load(R / folder / fn):
            if r.get('mode') == 'video':
                out[(show, str(r['question_id']))] = r
    return out


runs = {'InternVL2 (documented input both)': (r2('InternVL2-8B', 'ivl_videofmt_validopt'),
                                              r2('InternVL2-8B', 'pair_oracle_validopt')),
        'Qwen': (original('Qwen25VL'), r2('Qwen25VL', 'pair_oracle_validopt'))}
for m, (base, new) in runs.items():
    keys = [k for k in base if k in new and ok(base[k]) and ok(new[k])]
    units = defaultdict(list)
    for k in keys:
        units[ep_of.get(k, k[0] + k[1])].append((gender(base[k]), pc(new[k]) - pc(base[k]),
                                                  pc(base[k]), pc(new[k])))
    def gain(u, g):
        x = [d for gg, d, _, _ in u if gg == g]
        return 100 * sum(x) / len(x) if x else None
    def st(u):
        c, s = gain(u, 'cross'), gain(u, 'same')
        return None if c is None or s is None else c - s
    flat = [x for v in units.values() for x in v]
    for g in ('same', 'cross'):
        x = [x for x in flat if x[0] == g]
        print(f'  {m}, {g:5s}: n={len(x)}  uniform {100*sum(a for *_, a, _ in x)/len(x):.1f}% -> '
              f'oracle {100*sum(b for *_, b in x)/len(x):.1f}%  gain {gain(flat, g):+.1f} pp')
    est, lo, hi, p, ne = boot(units, st)
    print(f'  {m}: gain cross - gain same = {est:+.1f} pp, episode 95% CI [{lo:+.1f}, {hi:+.1f}], '
          f'bootstrap p={p:.4f}  ({len(keys)} pairs, {ne} episodes)\n')
