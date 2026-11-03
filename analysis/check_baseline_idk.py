"""How often each model chooses the "I don't know" option in the standard five-option
prompt (its position varies by question). Read-only.
"""
import ast
import json
from pathlib import Path

ANNOT = Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json')
RESULTS = Path(r'D:\eval_workspace\results\global_appearance')

MODELS = {
    'InternVL2-8B': RESULTS / 'InternVL2-8B' / 'predictions.json',
    'Qwen2.5-VL-7B': RESULTS / 'Qwen25VL' / 'predictions.json',
    'LLaVA-NeXT-Video-7B': RESULTS / 'LLaVA-NeXT-Video-7B' / 'predictions.json',
    'Gemini 2.5 Flash': RESULTS / 'Gemini' / 'predictions.json',
}


def find_idk_letter(options):
    """Return the letter (A-E) corresponding to the 'I don't know' option,
    or None if not present in the options list.
    """
    for i, opt in enumerate(options):
        # IDK is sometimes ["I don't know"] (a 1-element list) or just the string
        if isinstance(opt, list):
            for item in opt:
                if "don't know" in str(item).lower() or "i dont know" in str(item).lower():
                    return chr(ord('A') + i)
        elif isinstance(opt, str):
            if "don't know" in opt.lower() or "i dont know" in opt.lower():
                return chr(ord('A') + i)
    return None


# Load annotations
print('Loading annotations...')
with open(ANNOT) as f:
    annots = json.load(f)
annot_by_qid = {str(a['question_id']): a for a in annots}
print(f'  {len(annots)} questions\n')


# For each model
for model_name, pred_path in MODELS.items():
    if not pred_path.exists():
        print(f'{model_name}: predictions file not found at {pred_path}')
        continue

    with open(pred_path) as f:
        preds = json.load(f)

    total = 0
    questions_with_idk = 0
    picked_idk = 0
    idk_letter_distribution = {}

    for r in preds:
        qid = str(r.get('question_id', ''))
        annot = annot_by_qid.get(qid)
        if not annot:
            continue

        try:
            options = ast.literal_eval(annot.get('options', '[]'))
        except (SyntaxError, ValueError):
            continue

        # Predicted letter
        pred_letter = r.get('predicted_letter', r.get('answer', r.get('prediction', '')))
        if not pred_letter or pred_letter not in 'ABCDE':
            continue

        total += 1

        # Find the IDK letter for this question
        idk_letter = find_idk_letter(options)
        if idk_letter is None:
            continue

        questions_with_idk += 1
        idk_letter_distribution[idk_letter] = idk_letter_distribution.get(idk_letter, 0) + 1

        if pred_letter == idk_letter:
            picked_idk += 1

    pct_of_total = picked_idk / total * 100 if total else 0
    pct_of_eligible = picked_idk / questions_with_idk * 100 if questions_with_idk else 0

    print(f'=== {model_name} ===')
    print(f'  Total predictions:                {total}')
    print(f'  Questions where IDK was an option: {questions_with_idk}/{total}')
    print(f'    IDK letter distribution:        {sorted(idk_letter_distribution.items())}')
    print(f'  Model picked the IDK option:      {picked_idk}')
    print(f'    As % of all predictions:        {pct_of_total:.2f}%')
    print(f'    As % of IDK-eligible questions: {pct_of_eligible:.2f}%')
    print()


print('=' * 70)
print('Interpretation:')
print('  If "% of IDK-eligible" is LOW (< 5%), models rarely abstain via the')
print('  existing IDK option. Adding option F is meaningful.')
print('  If "% of IDK-eligible" is HIGH (> 15%), the existing IDK already')
print('  captures abstention. Option F may be partly redundant.')
