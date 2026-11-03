"""LLaVA-NeXT-Video-7B with black frames on BBT (129 questions): 16 black 224x224 frames,
the standard multiple-choice prompt, greedy decoding, at most 10 new tokens. Writes
predictions_textonly_v2.json. Resumable.
"""
import os, json, ast, sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch

MODEL_PATH = r'D:\models\llava_next_video_7b'
VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
OUT_DIR = Path(r'D:\eval_workspace\results\global_appearance\LLaVA-NeXT-Video-7B')
OLD_FILE = OUT_DIR / 'predictions_textonly.json'
OUT_FILE = OUT_DIR / 'predictions_textonly_v2.json'

# ---- the same 129 questions as the old text-only file ----
old = json.load(open(OLD_FILE, encoding='utf-8'))
qids = [str(r['question_id']) for r in old]
by_id = {}
for ann in (VAL_ANN, TR_ANN):
    for d in json.load(open(ann, encoding='utf-8')):
        by_id.setdefault(str(d['question_id']), d)
items = [by_id[q] for q in qids if q in by_id]
print(f'{len(qids)} question ids in old file, {len(items)} found in annotations')
if len(items) != len(qids):
    sys.exit('Some question ids were not found -- check the input file.')


def parse_options(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return eval(s)


def build_mc_prompt(question, item):          # standard multiple-choice prompt (Appendix B)
    label_chars = 'ABCDE'
    opts = parse_options(item['options'])
    options_text = ''
    for i, opt in enumerate(opts):
        if i >= len(label_chars):
            break
        text = ', '.join(opt) if isinstance(opt, list) else str(opt)
        options_text += f'\n{label_chars[i]}. {text}'
    return (
        f"Watch the full video carefully and track each character's appearance.\n\n"
        f"Question: {question}\n"
        f"Options:{options_text}\n\n"
        f"Reply with ONLY a single letter: A, B, C, D, or E."
    )


print('Loading LLaVA-NeXT-Video-7B...')
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
processor = LlavaNextVideoProcessor.from_pretrained(MODEL_PATH)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
model.eval()


def llava_run(frames, prompt_text, max_new_tokens=10):
    np_frames = np.stack([np.array(f) for f in frames])
    conv = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt_text}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[np_frames], padding=True,
                       return_tensors='pt').to('cuda', torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


black_frames = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(16)]
results = json.load(open(OUT_FILE, encoding='utf-8')) if OUT_FILE.exists() else []
done = {r['question_id'] for r in results}
for k, item in enumerate(items, 1):
    qid = str(item['question_id'])
    if qid in done:
        continue
    try:
        resp = llava_run(black_frames, build_mc_prompt(item['question'], item))
        pred = next((c for c in resp.upper() if c in 'ABCDE'), 'X')
    except Exception as e:
        print(f'  ERROR Q{qid}: {e}')
        resp, pred = f'ERROR: {e}', 'X'
    results.append({'question_id': qid, 'correct_label': item['answer'], 'prediction': pred,
                    'correct': pred == item['answer'], 'raw_response': resp,
                    'prompt': 'standard MC prompt', 'frames': '16 black 224x224'})
    json.dump(results, open(OUT_FILE, 'w', encoding='utf-8'), indent=2)
    print(f'[{k}/{len(items)}] Q{qid} GT={item["answer"]} Pred={pred}')
    torch.cuda.empty_cache()

# ---- summary ----
n = len(results)
c = sum(r['correct'] for r in results)
bad = sum(r['prediction'] == 'X' for r in results)
old_by = {str(r['question_id']): r['prediction'] for r in old}
same = sum(old_by.get(r['question_id']) == r['prediction'] for r in results)
dist = {L: sum(r['prediction'] == L for r in results) for L in 'ABCDEX'}
print('\n' + '=' * 60)
print(f'NEW LLaVA text-only: {c}/{n} = {100*c/n:.1f}%   (old file: '
      f'{sum(r["correct"] for r in old)}/{len(old)} = {100*sum(r["correct"] for r in old)/len(old):.1f}%)')
print(f'unparseable (X): {bad}')
print(f'same letter as old file: {same}/{n}')
print(f'letter distribution: {dist}')
print('=' * 60)
