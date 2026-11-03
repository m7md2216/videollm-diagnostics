"""Collects the open-ended answers of all four models into openended_v2_for_rejudge.json for
the LLM judge, with each question's multiple-choice result for the matched comparison.
"""
import json
from pathlib import Path

RES = Path(r'D:\eval_workspace\results')
GA = RES / 'global_appearance'
OUT = RES / 'openended_v2_for_rejudge.json'
NEW = {'LLaVA-NeXT-Video-7B': 'LLaVA-NeXT-Video-7B', 'InternVL2-8B': 'InternVL2-8B',
       'Qwen2.5-VL-7B': 'Qwen25VL'}

items = []
for model, folder in list(NEW.items()) + [('Gemini-2.5-Flash', 'Gemini')]:
    src = GA / folder / ('predictions_openended_v2.json' if model in NEW else 'predictions_openended_judged.json')
    old = {str(r['question_id']): r.get('judge_score')
           for r in json.load(open(GA / folder / 'predictions_openended_judged.json', encoding='utf-8'))}
    mc = {str(r['question_id']): bool(r.get('correct'))
          for r in json.load(open(GA / folder / 'predictions.json', encoding='utf-8'))}
    rows = json.load(open(src, encoding='utf-8'))
    hit = sum(1 for r in rows if r.get('hit_limit'))
    for r in rows:
        q = str(r['question_id'])
        items.append(dict(model=model, question_id=q, question=r.get('question', ''),
                          ground_truth=r.get('ground_truth', ''), model_answer=r.get('model_answer', ''),
                          old_score=old.get(q), mc_correct=mc.get(q)))
    print(f'  {model:22s} {len(rows)} answers from {src.name}'
          + (f'  ({hit} hit the token limit)' if model in NEW else ''))

human = []
kp = RES / 'human_validation' / 'kappa_results.json'
if kp.exists():
    for h in json.load(open(kp, encoding='utf-8')).get('paired_scores', []):
        if h.get('model') == 'Gemini-2.5-Flash':
            human.append(dict(model=h['model'], question_id=str(h['qid']), human_score=h['human']))
print(f'  human scores kept (Gemini only, answers unchanged): {len(human)}')

json.dump(dict(items=items, human=human), open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'\nWrote {len(items)} answers to {OUT}')
