"""Open-ended failure modes per model (fabrication, refusal, repetition, partial, other),
with mean judge score and fully correct count. Read-only.
"""
import json
import os
from collections import Counter
from pathlib import Path

R = Path(r'D:\eval_workspace\results\global_appearance')

FILES = [
    ('LLaVA-NeXT-Video-7B', R / 'LLaVA-NeXT-Video-7B' / 'predictions_openended_judged.json'),
    ('InternVL2-8B',        R / 'InternVL2-8B'        / 'predictions_openended_judged.json'),
    ('Qwen2.5-VL-7B',       R / 'Qwen25VL'            / 'predictions_openended_judged.json'),
    ('Gemini-2.5-Flash',    R / 'Gemini'              / 'predictions_openended_judged.json'),
]

print(f'{"Model":<24} {"n":<4} {"Eff %":<8} {"Fully":<8} {"F %":<7} {"R %":<7} {"X %":<7} {"P %":<7}')
print('-' * 80)

for name, path in FILES:
    if not path.exists():
        print(f'{name:<24} MISSING: {path}')
        continue
    with open(path) as f:
        data = json.load(f)
    n = len(data)
    scored = [r for r in data if r.get('judge_score') is not None]
    if not scored:
        print(f'{name:<24} NO SCORES')
        continue

    total = sum(r['judge_score'] for r in scored)
    eff = total / len(scored) * 100
    fully = sum(1 for r in scored if r['judge_score'] == 1.0)

    fm = Counter(r.get('failure_mode') for r in scored)
    pct = lambda k: fm.get(k, 0) / len(scored) * 100

    print(f'{name:<24} {n:<4} {eff:<8.1f} {fully:<8} '
          f'{pct("F"):<7.0f} {pct("R"):<7.0f} '
          f'{pct("X"):<7.0f} {pct("P"):<7.0f}')

print('\nKey: F=fabrication, R=refusal, X=repetition, P=partial match (got at least one outfit)')
