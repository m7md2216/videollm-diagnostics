"""Counts which name-swap pairs can enter the valid-option swap (no model calls).

A pair X -> Y is usable when Y has their own global-appearance question in the same
episode, so Y's correct outfit sequence is known and can replace one distractor while X's
correct answer stays in the option set. Writes valid_option_candidates.json.
"""
import json, ast, re
from pathlib import Path
from collections import defaultdict, Counter

ANN_FILES = {
    'val':   Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'),
    'train': Path(r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'),
}
RES = Path(r'D:\eval_workspace\results\global_appearance')
OUT = RES / 'valid_option_candidates.json'
SWAP_MODELS = {'InternVL2-8B': 'InternVL2-8B', 'Qwen2.5-VL-7B': 'Qwen25VL',
               'LLaVA-NeXT-Video-7B': 'LLaVA-NeXT-Video-7B'}

CAST = {
    'bbt': ['sheldon', 'leonard', 'howard', 'raj', 'penny', 'bernadette', 'amy'],
    'friends': ['rachel', 'monica', 'phoebe', 'ross', 'chandler', 'joey'],
    'castle': ['castle', 'beckett', 'alexis', 'martha', 'esposito', 'ryan', 'lanie'],
}
FEMALE = {'penny', 'bernadette', 'amy', 'rachel', 'monica', 'phoebe',
          'beckett', 'alexis', 'martha', 'lanie'}

def show_of(path):
    p = path.lower()
    for s in CAST:
        if f'/{s}/' in p.replace('\\', '/'):
            return s
    return None

def char_of(question, show):
    q = question.lower()
    # 'castle' also appears in other words/contexts; require whole word
    hits = [c for c in CAST.get(show, []) if re.search(rf'\b{c}\b', q)]
    return hits[0] if len(hits) == 1 else (None if not hits else 'MULTI')

def parse_opts(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def norm_seq(seq):
    if isinstance(seq, list):
        return tuple(str(x).strip().lower() for x in seq)
    return (str(seq).strip().lower(),)

def is_idk(opt):
    return any("don't know" in str(x).lower() or 'do not know' in str(x).lower()
               for x in (opt if isinstance(opt, list) else [opt]))

# ---------------------------------------------------------------- load annotations
items = {}          # qid -> dict
by_ep_char = defaultdict(list)
unmatched = Counter()
for split, p in ANN_FILES.items():
    if not p.exists():
        print(f'  MISSING annotation file: {p}'); continue
    data = json.load(open(p, encoding='utf-8'))
    for x in data:
        show = show_of(x.get('video_path_mp4', ''))
        if show is None:
            continue
        ch = char_of(x.get('question', ''), show)
        opts = parse_opts(x.get('options', ''))
        try:
            gi = int(x.get('answer_idx'))
        except Exception:
            gi = None
        gold = opts[gi] if (opts and gi is not None and gi < len(opts)) else None
        q = str(x['question_id'])
        rec = dict(qid=q, split=split, show=show, ep=x['video_path_mp4'].lower().replace('\\', '/'),
                   char=ch, opts=opts, gold=gold, answer=x.get('answer'))
        items[q] = rec
        if ch in (None, 'MULTI'):
            unmatched[(show, ch)] += 1
            continue
        if gold is not None and not is_idk(gold):
            by_ep_char[(rec['ep'], ch)].append(rec)
    print(f'  loaded {split}: {len(data)} questions')

print(f'  questions without exactly one main-cast name (not usable as targets): '
      f'{dict(unmatched)}')

# ---------------------------------------------------------------- helper
def build_candidate(src, tgt_char):
    """Return (status, info) for swapping src's character to tgt_char."""
    cands = by_ep_char.get((src['ep'], tgt_char), [])
    if not cands:
        return 'no_target_question', None
    tgt = cands[0]  # first question about Y in the same episode
    ygold = norm_seq(tgt['gold'])
    xgold = norm_seq(src['gold'])
    if ygold == xgold:
        return 'same_sequence_as_X', None
    opt_norm = [norm_seq(o) for o in src['opts']]
    already = ygold in opt_norm
    nx = len(src['gold']) if isinstance(src['gold'], list) else 1
    ny = len(tgt['gold']) if isinstance(tgt['gold'], list) else 1
    return ('already_in_options' if already else 'needs_insertion'), dict(
        target_qid=tgt['qid'], target_split=tgt['split'],
        y_gold=tgt['gold'], y_n_outfits=ny, x_n_outfits=nx,
        y_already_at=(opt_norm.index(ygold) if already else None))

# ---------------------------------------------------------------- PART 1: existing BBT swap pairs
print('\n' + '=' * 70)
print('PART 1. BBT SWAP PAIRS')
print('=' * 70)
existing = {}
for name, f in SWAP_MODELS.items():
    p = RES / f / 'predictions_name_swap_full.json'
    if not p.exists():
        print(f'  missing {p}'); continue
    for r in json.load(open(p, encoding='utf-8')):
        existing.setdefault(name, {})[str(r['question_id'])] = (
            str(r.get('original_char', '')).lower(), str(r.get('swap_char', '')).lower())

names = list(existing)
if len(names) > 1:
    base = existing[names[0]]
    for n in names[1:]:
        same = sum(1 for q in base if existing[n].get(q) == base[q])
        print(f'  identical swap pairs {names[0]} vs {n}: {same}/{len(base)}')

per_model = {}
for name, pairs in existing.items():
    st = Counter()
    rows = []
    for q, (xc, yc) in pairs.items():
        src = items.get(q)
        if not src or src['gold'] is None or src['opts'] is None:
            st['no_annotation'] += 1; continue
        status, info = build_candidate(src, yc)
        st[status] += 1
        if info:
            rows.append(dict(qid=q, x=xc, y=yc, status=status, **info))
    per_model[name] = rows
    usable = st['needs_insertion'] + st['already_in_options']
    print(f'\n  {name}: {len(pairs)} pairs')
    for k in ('needs_insertion', 'already_in_options', 'no_target_question',
              'same_sequence_as_X', 'no_annotation'):
        print(f'    {k:22s} {st[k]}')
    print(f'    -> USABLE with the same Y: {usable}/{len(pairs)}')
    if rows:
        lm = sum(1 for r in rows if r['y_n_outfits'] == r['x_n_outfits'])
        print(f'       of which Y and X have the same number of outfits: {lm}/{len(rows)}')
    if st['already_in_options']:
        print(f'    NOTE: {st["already_in_options"]} pairs ALREADY had Y\'s correct sequence '
              f'among the original options -> can be scored today, no new run.')

# ---------------------------------------------------------------- PART 2: re-targeting
print('\n' + '=' * 70)
print('PART 2. IF WE PICK Y FRESH (any other main-cast character with a question in the episode)')
print('=' * 70)
retarget = []
for show in CAST:
    pool = [r for r in items.values() if r['show'] == show and r['char'] not in (None, 'MULTI')
            and r['gold'] is not None and r['opts'] is not None and not is_idk(r['gold'])]
    n_any, n_same, n_cross, n_lenmatch = 0, 0, 0, 0
    for src in pool:
        opts_here = []
        for yc in CAST[show]:
            if yc == src['char']:
                continue
            status, info = build_candidate(src, yc)
            if info:
                opts_here.append((yc, status, info))
        if opts_here:
            n_any += 1
            gx = src['char'] in FEMALE
            if any((yc in FEMALE) == gx for yc, _, _ in opts_here): n_same += 1
            if any((yc in FEMALE) != gx for yc, _, _ in opts_here): n_cross += 1
            if any(i['y_n_outfits'] == i['x_n_outfits'] for _, _, i in opts_here): n_lenmatch += 1
            retarget.append(dict(qid=src['qid'], split=src['split'], show=show, x=src['char'],
                                 targets=[dict(y=yc, status=s, **i) for yc, s, i in opts_here]))
    splits = Counter(r['split'] for r in pool)
    print(f'  {show:8s} questions with a main-cast name: {len(pool):4d} {dict(splits)}')
    print(f'           with >=1 valid Y in same episode: {n_any:4d}   '
          f'(same-gender Y available: {n_same}, cross-gender: {n_cross}, '
          f'outfit-count match: {n_lenmatch})')

json.dump(dict(existing_pairs=per_model, retarget=retarget), open(OUT, 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print(f'\nWrote {OUT}')
