"""Name swap by gender relation: for same- and cross-gender swaps, where the answer goes
after the swap (unchanged, the original character's correct option, or another option).
Read-only.
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS = Path(r'D:\eval_workspace\results\global_appearance')

MALE = {'sheldon', 'leonard', 'howard', 'raj'}
FEMALE = {'penny', 'bernadette', 'amy'}


def gender(name):
    n = name.lower()
    if n in MALE:
        return 'M'
    if n in FEMALE:
        return 'F'
    return '?'


for model_dir in ['LLaVA-NeXT-Video-7B', 'InternVL2-8B', 'Qwen25VL']:
    swap_file = RESULTS / model_dir / 'predictions_name_swap_full.json'
    if not swap_file.exists():
        print(f'{model_dir}: missing {swap_file.name}')
        continue

    with open(swap_file) as f:
        records = json.load(f)

    same_gen = {'n': 0, 'orig_correct': 0, 'swap_correct': 0, 'answer_changed': 0}
    cross_gen = {'n': 0, 'orig_correct': 0, 'swap_correct': 0, 'answer_changed': 0}

    for r in records:
        g_orig = gender(r['original_char'])
        g_swap = gender(r['swap_char'])
        if g_orig == '?' or g_swap == '?':
            continue

        bucket = same_gen if g_orig == g_swap else cross_gen
        bucket['n'] += 1
        if r['orig_correct']:
            bucket['orig_correct'] += 1
        # 'swap correct' means: after the swap, did the model pick the ORIGINAL question's
        # correct answer? This measures whether the model's swap prediction lands on
        # the right letter for the original character (we can't measure right-for-swapped-character).
        if r['swap_pred'] == r['correct_label']:
            bucket['swap_correct'] += 1
        if r['answer_changed']:
            bucket['answer_changed'] += 1

    def pct(num, den):
        return f'{num/den*100:.1f}%' if den > 0 else 'N/A'

    print(f'\n=== {model_dir} (n={len(records)}) ===')
    print(f'                       Same-gender swap    Cross-gender swap')
    print(f'  N:                       {same_gen["n"]:<20}{cross_gen["n"]}')
    print(f'  Orig correct:            {pct(same_gen["orig_correct"], same_gen["n"]):<20}{pct(cross_gen["orig_correct"], cross_gen["n"])}')
    print(f'  Swap matches orig GT:    {pct(same_gen["swap_correct"], same_gen["n"]):<20}{pct(cross_gen["swap_correct"], cross_gen["n"])}')
    print(f'  Answer changed:          {pct(same_gen["answer_changed"], same_gen["n"]):<20}{pct(cross_gen["answer_changed"], cross_gen["n"])}')

    # The interesting question: among pairs where answer DID change after cross-gender swap,
    # did it tend to LAND on the original correct answer?
    cross_changed = [r for r in records
                     if r['answer_changed']
                     and gender(r['original_char']) != gender(r['swap_char'])
                     and gender(r['original_char']) != '?' and gender(r['swap_char']) != '?']
    cross_changed_now_correct = sum(1 for r in cross_changed if r['swap_pred'] == r['correct_label'])
    cross_changed_was_correct = sum(1 for r in cross_changed if r['orig_pred'] == r['correct_label'])
    print(f'  Cross-gender changes:    {len(cross_changed)}')
    if cross_changed:
        print(f'    Was correct, became wrong:   {cross_changed_was_correct - sum(1 for r in cross_changed if r["orig_pred"] == r["correct_label"] and r["swap_pred"] == r["correct_label"])}')
        print(f'    Was wrong, became correct:   {cross_changed_now_correct - sum(1 for r in cross_changed if r["orig_pred"] == r["correct_label"] and r["swap_pred"] == r["correct_label"])}')
        print(f'    Was wrong, became wrong:     {sum(1 for r in cross_changed if r["orig_pred"] != r["correct_label"] and r["swap_pred"] != r["correct_label"])}')

print('\nNote: "Swap matches orig GT" means the post-swap prediction happens to be the')
print('correct letter for the ORIGINAL question. We cannot measure correctness against')
print('the swapped character because they typically have no ground truth in the episode.')
