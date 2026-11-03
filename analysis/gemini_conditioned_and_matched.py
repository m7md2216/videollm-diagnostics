"""Gemini name-swap comparison (read-only).

Part 1: conditioned answer-change rate on the 20 parseable pairs and the gender split.
Part 2: all four models on the same question IDs: the 20 Gemini pairs with exact McNemar
        tests against each open-source model, and the 14 pairs where every model received
        the identical swap.
"""
import json
from math import comb, sqrt
from pathlib import Path

RES = Path(r'D:\eval_workspace\results\global_appearance')
ANN = Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json')
FOLDERS = {'InternVL2-8B': 'InternVL2-8B', 'Qwen2.5-VL-7B': 'Qwen25VL',
           'LLaVA-NeXT-Video-7B': 'LLaVA-NeXT-Video-7B', 'Gemini-2.5-Flash': 'Gemini'}
FEMALE = {'penny', 'bernadette', 'amy'}
MALE = {'leonard', 'sheldon', 'howard', 'raj'}
LET = set('ABCDE')

def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))

def fmt(k, n):
    if n == 0:
        return 'n=0'
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100*k/n:5.1f}%  [{lo:4.1f}, {hi:4.1f}]'

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def get(r, *names):
    for nm in names:
        if nm in r and r[nm] is not None:
            return r[nm]
    return None

def gender(name):
    n = str(name).strip().lower()
    return 'F' if n in FEMALE else 'M' if n in MALE else '?'

gold = {str(x['question_id']): x.get('answer')
        for x in json.load(open(ANN, encoding='utf-8'))}

# load every model's swap file into {qid: record}
swaps = {}
for name, f in FOLDERS.items():
    p = RES / f / 'predictions_name_swap_full.json'
    if not p.exists():
        print(f'  MISSING: {p}'); continue
    recs = {}
    for r in json.load(open(p, encoding='utf-8')):
        q = str(r['question_id'])
        recs[q] = dict(
            orig=get(r, 'orig_pred', 'original_pred', 'pred_original'),
            swap=get(r, 'swap_pred', 'swapped_pred', 'pred_swap'),
            oc=get(r, 'original_char', 'orig_char'),
            sc=get(r, 'swap_char', 'swapped_char'),
        )
    swaps[name] = recs
    none_fields = sum(1 for v in recs.values() if v['orig'] is None or v['swap'] is None)
    print(f'  loaded {name:22s} n={len(recs)}'
          + (f'   WARNING {none_fields} records missing pred fields' if none_fields else ''))

def valid(v):
    return v['orig'] in LET and v['swap'] in LET

def changed(v):
    return v['orig'] != v['swap']

# ------------------------------------------------------------------ PART 1
print('\n' + '=' * 70)
print('PART 1. GEMINI ON THE 20 VALID PAIRS')
print('=' * 70)
G = swaps['Gemini-2.5-Flash']
gv = {q: v for q, v in G.items() if valid(v)}
print(f'  valid pairs: {len(gv)} (excluded: '
      f'{sorted(q for q in G if q not in gv)})')
print(f'  unconditioned : {fmt(sum(changed(v) for v in gv.values()), len(gv))}')

cond = {q: v for q, v in gv.items() if v['orig'] == gold.get(q)}
print(f'  orig-correct  : {len(cond)} of {len(gv)} valid pairs')
print(f'  CONDITIONED   : {fmt(sum(changed(v) for v in cond.values()), len(cond))}')

for label, subset in (('all valid', gv), ('orig-correct', cond)):
    same = [v for v in subset.values() if gender(v['oc']) == gender(v['sc']) != '?']
    cross = [v for v in subset.values()
             if '?' not in (gender(v['oc']), gender(v['sc'])) and gender(v['oc']) != gender(v['sc'])]
    unk = [v for v in subset.values() if '?' in (gender(v['oc']), gender(v['sc']))]
    print(f'  {label:13s} same-gender : {fmt(sum(map(changed, same)), len(same))}')
    print(f'  {label:13s} cross-gender: {fmt(sum(map(changed, cross)), len(cross))}'
          + (f'   ({len(unk)} unknown-gender names)' if unk else ''))

print('\n  Per-record (orig-correct marked *):')
for q, v in sorted(gv.items()):
    star = '*' if q in cond else ' '
    print(f"   {star} Q{q}  {v['oc']:>10} -> {v['sc']:<10} gold={gold.get(q)}  "
          f"{v['orig']} -> {v['swap']}  changed={changed(v)}")

# ------------------------------------------------------------------ PART 2a
print('\n' + '=' * 70)
print('PART 2a. ALL MODELS ON THE SAME QUESTIONS AS GEMINI')
print('=' * 70)
for name, recs in swaps.items():
    if name == 'Gemini-2.5-Flash':
        continue
    shared = [q for q in gv if q in recs and valid(recs[q])]
    same_swap = [q for q in shared
                 if str(recs[q]['sc']).lower() == str(gv[q]['sc']).lower()
                 and str(recs[q]['oc']).lower() == str(gv[q]['oc']).lower()]
    print(f'\n  {name}')
    print(f'    Gemini QIDs present & valid : {len(shared)}/20   '
          f'identical swap (same orig->swap names): {len(same_swap)}')
    for lab, qs in (('same QIDs', shared), ('same QIDs + same swap', same_swap)):
        if not qs:
            print(f'    {lab:22s}: none'); continue
        m = sum(changed(recs[q]) for q in qs)
        g = sum(changed(gv[q]) for q in qs)
        b = sum(1 for q in qs if changed(gv[q]) and not changed(recs[q]))
        c = sum(1 for q in qs if changed(recs[q]) and not changed(gv[q]))
        print(f'    {lab:22s}: model {fmt(m, len(qs))}   Gemini {fmt(g, len(qs))}')
        print(f'    {"":22s}  discordant: Gemini-only {b}, model-only {c}, '
              f'exact McNemar p = {mcnemar_exact(b, c):.4f}')

# ------------------------------------------------------------------ PART 2b
print('\n' + '=' * 70)
print('PART 2b. CONDITIONED, ON QUESTIONS ALL 3 OPEN-SOURCE MODELS GOT RIGHT')
print('=' * 70)
OS = [n for n in swaps if n != 'Gemini-2.5-Flash']
common = None
for n in OS:
    ok = {q for q, v in swaps[n].items() if valid(v) and v['orig'] == gold.get(q)}
    print(f'  {n:22s} orig-correct n = {len(ok)}')
    common = ok if common is None else common & ok
common = sorted(common or [])
print(f'  intersection: {len(common)} questions')
for n in OS:
    k = sum(changed(swaps[n][q]) for q in common)
    print(f'    {n:22s} {fmt(k, len(common))}')
if 'Gemini-2.5-Flash' in swaps:
    gq = [q for q in common if q in gv and gv[q]['orig'] == gold.get(q)]
    print(f'    {"Gemini (also correct)":22s} {fmt(sum(changed(gv[q]) for q in gq), len(gq))}')
