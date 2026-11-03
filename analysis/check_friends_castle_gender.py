"""Friends and Castle name swaps (124 pairs per show, three models): same/cross-gender
split, and which pairs can enter the valid-option swap (the new character has their own
question in the same episode). Writes valid_option_candidates_fc.json. Read-only otherwise.
"""
import json, ast, re
from pathlib import Path
from collections import defaultdict, Counter
from math import comb

RES = Path(r'D:\eval_workspace\results\global_appearance')
ANN = {'val': Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'),
       'train': Path(r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json')}
MODELS = {'InternVL2': 'InternVL2-8B', 'Qwen': 'Qwen25VL', 'LLaVA': 'LLaVA-NeXT-Video-7B'}
CAST = {'friends': ['rachel', 'monica', 'phoebe', 'ross', 'chandler', 'joey'],
        'castle': ['castle', 'beckett', 'alexis', 'martha', 'esposito', 'ryan', 'lanie']}
FEMALE = {'rachel', 'monica', 'phoebe', 'beckett', 'alexis', 'martha', 'lanie'}


def fisher_two_sided(a, b, c, d):
    """Fisher exact test on [[a,b],[c,d]]."""
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    p0 = comb(n1, a) * comb(n2, c) / tot
    p = 0.0
    for x in range(max(0, k - n2), min(k, n1) + 1):
        px = comb(n1, x) * comb(n2, k - x) / tot
        if px <= p0 * (1 + 1e-9):
            p += px
    return min(p, 1.0)


def pct(k, n):
    return f'{k}/{n} = {100*k/n:5.1f}%' if n else '  -  '


# ------------------------------------------------------------------ PART A
print('=' * 78)
print('PART A. Existing Friends/Castle name swaps, split by gender')
print('=' * 78)
pairs_by_show = {}
for show in ('friends', 'castle'):
    for m, folder in MODELS.items():
        f = RES / folder / f'predictions_name_swap_{show}_expanded.json'
        if not f.exists():
            print(f'  missing {f}'); continue
        rows = json.load(open(f, encoding='utf-8'))
        st = Counter()
        for r in rows:
            x = str(r.get('original_name') or r.get('original_char') or '').lower()
            y = str(r.get('swap_name') or r.get('swap_char') or '').lower()
            pairs_by_show.setdefault(show, {})[str(r['question_id'])] = (x, y)
            if x not in CAST[show] or y not in CAST[show]:
                st['unknown_name'] += 1; continue
            g = 'same' if (x in FEMALE) == (y in FEMALE) else 'cross'
            st[g + '_n'] += 1; st[g + '_chg'] += bool(r['answer_changed'])
            if r.get('orig_correct'):
                st[g + '_cn'] += 1; st[g + '_cchg'] += bool(r['answer_changed'])
        p = fisher_two_sided(st['same_chg'], st['same_n'] - st['same_chg'],
                             st['cross_chg'], st['cross_n'] - st['cross_chg'])
        print(f'\n  {show.upper():8s} {m:10s} n={len(rows)}  (unknown names: {st["unknown_name"]})')
        print(f'    answer changed  same-gender {pct(st["same_chg"], st["same_n"])}   '
              f'cross-gender {pct(st["cross_chg"], st["cross_n"])}   Fisher p={p:.3g}')
        print(f'    among originally-correct: same {pct(st["same_cchg"], st["same_cn"])}   '
              f'cross {pct(st["cross_cchg"], st["cross_cn"])}')

# ------------------------------------------------------------------ PART B
print('\n' + '=' * 78)
print('PART B. Valid-option feasibility for the same Friends/Castle pairs')
print('=' * 78)


def show_of(path):
    p = path.lower().replace('\\', '/')
    return next((s for s in CAST if f'/{s}/' in p), None)


def char_of(q, show):
    hits = [c for c in CAST[show] if re.search(rf'\b{c}\b', q.lower())]
    return hits[0] if len(hits) == 1 else None


def opts_of(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def norm(seq):
    return tuple(str(x).strip().lower() for x in (seq if isinstance(seq, list) else [seq]))


items, by_ep_char = {}, defaultdict(list)
for split, p in ANN.items():
    for x in json.load(open(p, encoding='utf-8')):
        show = show_of(x.get('video_path_mp4', ''))
        if show is None:
            continue
        opts = opts_of(x.get('options', ''))
        try:
            gold = opts[int(x['answer_idx'])]
        except Exception:
            gold = None
        rec = dict(qid=str(x['question_id']), split=split, show=show,
                   ep=x['video_path_mp4'].lower().replace('\\', '/'),
                   char=char_of(x.get('question', ''), show), opts=opts, gold=gold)
        items[rec['qid']] = rec
        if rec['char'] and gold is not None and "know" not in str(gold).lower():
            by_ep_char[(rec['ep'], rec['char'])].append(rec)

out = {}
for show, pairs in pairs_by_show.items():
    st, rows = Counter(), []
    for q, (x, y) in pairs.items():
        src = items.get(q)
        if not src or src['gold'] is None or src['opts'] is None:
            st['no_annotation'] += 1; continue
        cands = by_ep_char.get((src['ep'], y), [])
        if not cands:
            st['no_Y_question'] += 1; continue
        tgt = cands[0]
        if norm(tgt['gold']) == norm(src['gold']):
            st['same_sequence'] += 1; continue
        nx = len(src['gold']) if isinstance(src['gold'], list) else 1
        ny = len(tgt['gold']) if isinstance(tgt['gold'], list) else 1
        g = 'same' if (x in FEMALE) == (y in FEMALE) else 'cross'
        st['usable'] += 1; st['usable_' + g] += 1; st['len_match'] += (nx == ny)
        rows.append(dict(qid=q, x=x, y=y, gender=g, target_qid=tgt['qid'],
                         y_gold=tgt['gold'], x_n_outfits=nx, y_n_outfits=ny))
    out[show] = rows
    print(f'\n  {show.upper()}: {len(pairs)} pairs -> usable {st["usable"]} '
          f'(same-gender {st["usable_same"]}, cross-gender {st["usable_cross"]}; '
          f'same outfit count {st["len_match"]})')
    print(f'    not usable: no Y question {st["no_Y_question"]}, same sequence {st["same_sequence"]}, '
          f'no annotation {st["no_annotation"]}')

OUT = RES / 'valid_option_candidates_fc.json'
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'\nWrote {OUT}')
