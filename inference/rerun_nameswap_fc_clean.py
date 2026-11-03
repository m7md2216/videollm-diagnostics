"""Friends/Castle name swap with full-name replacement, three models.

The replacement replaces the character's full name and title (e.g. "Dr. Ross Geller" ->
"Phoebe"), so the swapped question names an existing character. Only the swapped question
is re-asked; the original-question answer comes from the earlier run on the same pairs,
with the same frames and inference code. Pairs whose question names two cast members are
skipped. Writes <model>/predictions_name_swap_<show>_clean.json. Resumable.
"""
import os, json, ast, re, sys
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')

INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH = r'D:\models\qwen25_vl_7b'
LLAVA_PATH = r'D:\models\llava_next_video_7b'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
RES = Path(r'D:\eval_workspace\results\global_appearance')
FOLDERS = {'InternVL2-8B': 'InternVL2-8B', 'Qwen2.5-VL-7B': 'Qwen25VL',
           'LLaVA-NeXT-Video-7B': 'LLaVA-NeXT-Video-7B'}
SHOWS = ['friends', 'castle']
CAST = {'friends': ['rachel', 'monica', 'phoebe', 'ross', 'chandler', 'joey'],
        'castle': ['castle', 'beckett', 'alexis', 'martha', 'lanie', 'gates', 'esposito', 'ryan']}
FIRST = {'castle': ['Richard', 'Rick'], 'beckett': ['Kate', 'Katherine', 'Katie'],
         'esposito': ['Javier'], 'ryan': ['Kevin'], 'gates': ['Victoria']}
LAST = {'ross': ['Geller'], 'monica': ['Geller'], 'rachel': ['Green'], 'chandler': ['Bing'],
        'joey': ['Tribbiani'], 'phoebe': ['Buffay'], 'lanie': ['Parish'], 'alexis': ['Castle'],
        'martha': ['Rodgers']}
TITLE = r"(?:(?:Dr|Mr|Mrs|Ms|Miss|Detective|Det|Captain|Capt|Officer|Lieutenant|Lt|Sergeant|Sgt)\.?\s+)?"


def name_pattern(tok):
    pat = TITLE
    if FIRST.get(tok):
        pat += r'(?:(?:' + '|'.join(FIRST[tok]) + r')\s+)?'
    pat += r'\b' + tok.title() + r'\b'
    if LAST.get(tok):
        pat += r'(?:\s+(?:' + '|'.join(LAST[tok]) + r')\b)?'
    return pat


def parse_options(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return eval(s)


def build_mc_prompt(question, item):          # identical to run_expanded_nameswap.py
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


def load_frames(vpath, n=16):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    frames = vr.get_batch(idx).numpy()
    return [Image.fromarray(frames[i]) for i in range(n)]


ann = {str(d['question_id']): d for d in json.load(open(TR_ANN, encoding='utf-8'))}

# ---------------------------------------------------------------- build jobs per model/show
jobs = {}
for mname, folder in FOLDERS.items():
    for show in SHOWS:
        old = json.load(open(RES / folder / f'predictions_name_swap_{show}_expanded.json', encoding='utf-8'))
        rows, skip = [], Counter()
        for r in old:
            qid = str(r['question_id'])
            x = str(r.get('original_name') or r.get('original_char') or '').lower()
            y = str(r.get('swap_name') or r.get('swap_char') or '').lower()
            item = ann.get(qid)
            if x not in CAST[show] or y not in CAST[show]:
                skip['name not in cast'] += 1; continue
            if item is None:
                skip['no annotation'] += 1; continue
            if r.get('orig_pred') in (None, 'X'):
                skip['no usable original answer'] += 1; continue
            named = [c for c in CAST[show] if re.search(rf'\b{c}\b', item['question'].lower())]
            if named != [x]:
                skip['question names a different or second cast member'] += 1; continue
            q_y = re.sub(name_pattern(x), y.title(), item['question'])
            if q_y == item['question']:
                skip['name not found'] += 1; continue
            rows.append(dict(qid=qid, x=x, y=y, item=item, q_y=q_y, orig_pred=r['orig_pred']))
        jobs[(mname, show)] = rows
        if mname == 'InternVL2-8B':
            print(f'{show.upper()}: {len(old)} pairs -> {len(rows)} kept   skipped: {dict(skip)}')
            for j in rows[:3]:
                print(f"    {j['item']['question']}\n      -> {j['q_y']}")
if '--dry' in sys.argv:
    sys.exit('Dry run only (no model loaded).')


def run_set(mname, run_fn):
    for show in SHOWS:
        out = RES / FOLDERS[mname] / f'predictions_name_swap_{show}_clean.json'
        results = json.load(open(out, encoding='utf-8')) if out.exists() else []
        done = {r['question_id'] for r in results}
        todo = [j for j in jobs[(mname, show)] if j['qid'] not in done]
        print(f'\n--- {mname} / {show}: done {len(done)}, remaining {len(todo)}')
        for j in todo:
            item = j['item']
            try:
                frames = load_frames(os.path.join(BASE_DIR, item['video_path_mp4']), 16)
                resp = run_fn(frames, build_mc_prompt(j['q_y'], item))
                swap_pred = next((c for c in resp.upper() if c in 'ABCDE'), 'X')
            except Exception as e:
                print(f"    ERROR Q{j['qid']}: {e}"); continue
            results.append(dict(
                question_id=j['qid'], original_name=j['x'], swap_name=j['y'],
                correct_label=item['answer'], orig_pred=j['orig_pred'], swap_pred=swap_pred,
                orig_correct=j['orig_pred'] == item['answer'],
                answer_changed=j['orig_pred'] != swap_pred, question_swap=j['q_y'], raw_swap=resp))
            print(f"    Q{j['qid']} {j['x']}->{j['y']} orig={j['orig_pred']} swap={swap_pred}")
            json.dump(results, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            torch.cuda.empty_cache()
        n = len(results); c = sum(r['answer_changed'] for r in results)
        print(f'  {mname} {show}: changed {c}/{n} = {100*c/max(n,1):.1f}%')


# ---------------------------------------------------------------- models (copied from run_expanded_nameswap.py)
print('\nLoading InternVL2-8B...')
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                       T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
tok = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)
m = AutoModel.from_pretrained(INTERNVL_PATH, torch_dtype=torch.float16, device_map='cuda:0',
                              trust_remote_code=True).eval()


def iv(frames, prompt):
    pv = torch.stack([transform(f) for f in frames]).half().cuda()
    return m.chat(tok, pv, f'Frame1: <image>\n{prompt}', dict(max_new_tokens=10, do_sample=False)).strip()


run_set('InternVL2-8B', iv)
del m, tok; torch.cuda.empty_cache()

print('\nLoading Qwen2.5-VL-7B...')
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
proc = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.float16,
                                                       device_map='cuda:0', trust_remote_code=True).eval()


def qw(frames, prompt):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': f} for f in frames] +
                 [{'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(messages)
    inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=10, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


run_set('Qwen2.5-VL-7B', qw)
del m, proc; torch.cuda.empty_cache()

print('\nLoading LLaVA-NeXT-Video-7B...')
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
proc = LlavaNextVideoProcessor.from_pretrained(LLAVA_PATH)
m = LlavaNextVideoForConditionalGeneration.from_pretrained(LLAVA_PATH, torch_dtype=torch.float16,
                                                           device_map='cuda:0', low_cpu_mem_usage=True).eval()


def ll(frames, prompt):
    np_frames = np.stack([np.array(f) for f in frames])
    conv = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
    p = proc.apply_chat_template(conv, add_generation_prompt=True)
    inputs = proc(text=[p], videos=[np_frames], padding=True, return_tensors='pt').to('cuda', torch.float16)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=10, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


run_set('LLaVA-NeXT-Video-7B', ll)
print('\nALL DONE.')
