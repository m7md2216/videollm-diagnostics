"""Counts swapped questions in which the replacement name kept the original character's
title or surname (e.g. "Dr. Phoebe Geller"), per show, and compares those pairs with the
others. Read-only.
"""
import json, re
from pathlib import Path
from collections import Counter

RES = Path(r'D:\eval_workspace\results\global_appearance')
ANN = [r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json',
       r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json']
FIRST = {'castle': ['Richard', 'Rick'], 'beckett': ['Kate', 'Katherine', 'Katie'],
         'esposito': ['Javier'], 'ryan': ['Kevin'], 'gates': ['Victoria']}
LAST = {'ross': ['Geller'], 'monica': ['Geller'], 'rachel': ['Green'], 'chandler': ['Bing'],
        'joey': ['Tribbiani'], 'phoebe': ['Buffay'], 'lanie': ['Parish'], 'alexis': ['Castle'],
        'martha': ['Rodgers'], 'sheldon': ['Cooper'], 'leonard': ['Hofstadter'],
        'howard': ['Wolowitz'], 'raj': ['Koothrappali'], 'bernadette': ['Rostenkowski', 'Wolowitz'],
        'amy': ['Farrah Fowler', 'Fowler']}
TITLE = r"(?:Dr|Mr|Mrs|Ms|Miss|Detective|Det|Captain|Capt|Officer|Lieutenant|Lt|Sergeant|Sgt)\.?\s+"


def decorated(q, tok):
    """True if the name token appears with a title, first name or surname attached."""
    t = tok.title()
    if re.search(TITLE + r'(?:\w+\s+)?' + rf'\b{t}\b', q):
        return True
    if FIRST.get(tok) and re.search(r'\b(?:' + '|'.join(FIRST[tok]) + rf')\s+{t}\b', q):
        return True
    if LAST.get(tok) and re.search(rf'\b{t}\s+(?:' + '|'.join(LAST[tok]) + r')\b', q):
        return True
    return False


ann = {}
for p in ANN:
    for a in json.load(open(p, encoding='utf-8')):
        ann.setdefault(str(a['question_id']), a['question'])

SETS = [('BBT name swap', 'predictions_name_swap_full.json'),
        ('Friends name swap', 'predictions_name_swap_friends_expanded.json'),
        ('Castle name swap', 'predictions_name_swap_castle_expanded.json')]
for label, fn in SETS:
    print('\n' + '=' * 72); print(label); print('=' * 72)
    for model in ('InternVL2-8B', 'Qwen25VL', 'LLaVA-NeXT-Video-7B'):
        f = RES / model / fn
        if not f.exists():
            continue
        rows = json.load(open(f, encoding='utf-8'))
        st, ex = Counter(), []
        for r in rows:
            x = str(r.get('original_name') or r.get('original_char') or '').lower()
            q = ann.get(str(r['question_id']), '')
            d = decorated(q, x)
            st['dec' if d else 'clean'] += 1
            st[('dec' if d else 'clean') + '_chg'] += bool(r.get('answer_changed'))
            if d and len(ex) < 3:
                ex.append(q)
        nd, nc = st['dec'], st['clean']
        print(f"  {model:22s} title/full name: {nd}/{len(rows)}   answer changed: "
              f"with title/full name {st['dec_chg']}/{nd}  |  clean {st['clean_chg']}/{nc}")
        if model == 'InternVL2-8B':
            for q in ex:
                print(f'      e.g. {q}')

print('\n' + '=' * 72); print('BBT valid-option swap (question_y is stored in the file)'); print('=' * 72)
for model in ('InternVL2-8B', 'Qwen25VL'):
    f = RES / model / 'predictions_valid_option_swap.json'
    if not f.exists():
        continue
    rows = [r for r in json.load(open(f, encoding='utf-8')) if r['mode'] == 'video']
    dec = [r for r in rows if decorated(ann.get(str(r['question_id']), ''), r['original_char'])]
    pc = lambda rs: sum(r['pred_x'] == r['x_label'] and r['pred_y'] == r['y_label'] for r in rs)
    print(f"  {model:22s} title/full name: {len(dec)}/{len(rows)}   pair correct: "
          f"with {pc(dec)}/{len(dec)}  |  clean {pc([r for r in rows if r not in dec])}/{len(rows)-len(dec)}")
    for r in dec[:3]:
        print(f"      e.g. {r['question_y']}")
