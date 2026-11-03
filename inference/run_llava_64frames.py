"""LLaVA-NeXT-Video-7B with 64 frames on BBT validation (129 questions). Resumable.
"""
import os, json, ast, sys, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')

MODEL_PATH = r'D:\models\llava_next_video_7b'
BASE_DIR   = r'D:\InifiniBenchVideos\train_videos'
VAL_ANN    = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'

OUT_DIR  = Path(r'D:\eval_workspace\results\global_appearance\LLaVA-NeXT-Video-7B')
OUT_FILE = OUT_DIR / 'predictions_64frames.json'

# ---- Load BBT validation ----
with open(VAL_ANN) as f:
    val_data = [d for d in json.load(f) if d.get('is_visual', False)]
bbt_val = [d for d in val_data if 'bbt' in d['video_path_mp4'].lower()]
print(f'BBT val: {len(bbt_val)} questions\n')

# ---- Helpers ----
def parse_options(opts_field):
    try:
        return ast.literal_eval(opts_field)
    except:
        return eval(opts_field)

def build_mc_prompt(question, item):
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

def load_frames(vpath, n=64):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr)-1, n, dtype=int)
    frames = vr.get_batch(idx).numpy()
    return [Image.fromarray(frames[i]) for i in range(n)]

# ---- Load LLaVA model ----
print('Loading LLaVA-NeXT-Video-7B...')
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

processor = LlavaNextVideoProcessor.from_pretrained(MODEL_PATH)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True
)
model.eval()
print('Model loaded.\n')

def llava_run(frames, prompt_text):
    np_frames = np.stack([np.array(f) for f in frames])
    conv = [{'role':'user','content':[
        {'type':'video'},{'type':'text','text':prompt_text}
    ]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[np_frames],
                       padding=True, return_tensors='pt').to('cuda', torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

# ---- Run 64-frame BBT eval ----
done, results = set(), []
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        results = json.load(f)
    done = {str(r['question_id']) for r in results}

remaining = [d for d in bbt_val if str(d['question_id']) not in done]
print(f'Already done: {len(done)}, Remaining: {len(remaining)}')

start = time.time()
oom_count = 0
for item in remaining:
    qid   = str(item['question_id'])
    vpath = os.path.join(BASE_DIR, item['video_path_mp4'])
    prompt = build_mc_prompt(item['question'], item)
    try:
        frames = load_frames(vpath, 64)
        resp = llava_run(frames, prompt)
        pred = next((c for c in resp.upper() if c in 'ABCDE'), 'X')
    except torch.cuda.OutOfMemoryError as e:
        print(f'  OOM Q{qid}: GPU out of memory at 64 frames')
        oom_count += 1
        pred = 'OOM'
        resp = 'OOM'
        torch.cuda.empty_cache()
    except Exception as e:
        msg = str(e)[:100]
        print(f'  ERROR Q{qid}: {msg}')
        pred = 'X'
        resp = f'ERROR: {msg}'

    correct = pred == item['answer']
    results.append({
        'question_id':   qid,
        'correct_label': item['answer'],
        'prediction':    pred,
        'correct':       correct,
    })
    flag = 'OK' if correct else ('OOM' if pred == 'OOM' else '--')
    print(f'  [{flag}] Q{qid} | GT={item["answer"]} Pred={pred}')

    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    torch.cuda.empty_cache()

# Summary
elapsed = (time.time() - start) / 60
n = len(results)
valid = [r for r in results if r['prediction'] in 'ABCDE']
n_valid = len(valid)
n_correct = sum(r['correct'] for r in valid)
n_oom = sum(1 for r in results if r['prediction'] == 'OOM')

print(f'\n{"="*60}')
print(f'  LLaVA-NeXT-Video-7B @ 64 frames - BBT validation')
print(f'{"="*60}')
print(f'  Total rows:          {n}')
print(f'  Valid predictions:   {n_valid}')
print(f'  OOM failures:        {n_oom}')
if n_valid > 0:
    print(f'  Accuracy on valid:   {n_correct}/{n_valid} = {n_correct/n_valid*100:.1f}%')
if n_oom > 0:
    print(f'  NOTE: {n_oom} questions failed with OOM. Consistent with InternVL2 64f failure.')
print(f'  Elapsed: {elapsed:.1f} min')
print(f'  Output:  {OUT_FILE}')
