"""Open-ended answers that reach the 512-token limit: whether they repeat themselves or
are still listing new outfits. Read-only.
"""
import json, re
from pathlib import Path

RES = Path(r'D:\eval_workspace\results\global_appearance')
for folder in ('InternVL2-8B', 'Qwen25VL', 'LLaVA-NeXT-Video-7B'):
    f = RES / folder / 'predictions_openended_v2.json'
    if not f.exists():
        continue
    rows = json.load(open(f, encoding='utf-8'))
    long_ = [r for r in rows if r.get('hit_limit')]
    if not long_:
        continue
    loops, items_all = 0, []
    for r in long_:
        sents = [s.strip().lower() for s in re.split(r'[.\n]+', r['model_answer']) if len(s.strip()) > 15]
        uniq = len(set(sents)) / max(len(sents), 1)
        items = len(re.findall(r'(?m)^\s*\d+\.', r['model_answer']))
        r['_uniq'], r['_items'] = uniq, items
        items_all.append(items)
        if uniq < 0.6:
            loops += 1
    gt_n = [len(r['ground_truth'].split(',')) for r in long_]
    print(f'\n=== {folder}: {len(long_)} answers hit the limit')
    print(f'  looping (fewer than 60% distinct sentences): {loops}/{len(long_)}')
    print(f'  numbered items per answer: mean {sum(items_all)/len(items_all):.1f}, max {max(items_all)}'
          f'  | ground-truth outfits: mean {sum(gt_n)/len(gt_n):.1f}')
    for r in sorted(long_, key=lambda r: r['_uniq'])[:1] + sorted(long_, key=lambda r: -r['_uniq'])[:2]:
        print(f'\n  Q{r["question_id"]}  distinct-sentence ratio {r["_uniq"]:.2f}, {r["_items"]} numbered items')
        print(f'  GROUND TRUTH: {r["ground_truth"]}')
        print('  LAST 300 CHARS: ' + r['model_answer'][-300:].replace('\n', ' | '))
