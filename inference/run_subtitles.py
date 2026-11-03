"""Subtitle condition for InternVL2, Qwen and LLaVA on BBT validation (129 questions): the
dialogue from each episode's .srt file is added to the prompt (first 4,000 characters,
no speaker labels).
"""
import os, json, ast, re, sys, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')

INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH     = r'D:\models\qwen25_vl_7b'
LLAVA_PATH    = r'D:\models\llava_next_video_7b'

BASE_DIR      = r'D:\InifiniBenchVideos\train_videos'
SUBTITLE_BASE = r'D:\InifiniBenchVideos\train_videos'

VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'

OUT_DIR_INTERNVL = Path(r'D:\eval_workspace\results\global_appearance\InternVL2-8B')
OUT_DIR_QWEN     = Path(r'D:\eval_workspace\results\global_appearance\Qwen25VL')
OUT_DIR_LLAVA    = Path(r'D:\eval_workspace\results\global_appearance\LLaVA-NeXT-Video-7B')
for d in (OUT_DIR_INTERNVL, OUT_DIR_QWEN, OUT_DIR_LLAVA):
    d.mkdir(parents=True, exist_ok=True)

OUT_INTERNVL = OUT_DIR_INTERNVL / 'predictions_subtitle_v2.json'
OUT_QWEN     = OUT_DIR_QWEN     / 'predictions_subtitle_v2.json'
OUT_LLAVA    = OUT_DIR_LLAVA    / 'predictions_subtitle_v2.json'

# ---- Load BBT validation ----
with open(VAL_ANN) as f:
    val_data = [d for d in json.load(f) if d.get('is_visual', False)]
bbt_val = [d for d in val_data if 'bbt' in d['video_path_mp4'].lower()]
print(f'BBT val: {len(bbt_val)} questions\n')

# ---- Subtitle loader ----
def load_srt_dialogue(srt_relative_path, max_chars=4000):
    """Read .srt file, strip timestamps/indices/tags, return dialogue text only."""
    if not srt_relative_path:
        return ''
    full_path = os.path.join(SUBTITLE_BASE, srt_relative_path)
    if not os.path.exists(full_path):
        alt_path = os.path.join(r'D:\InifiniBenchVideos', srt_relative_path)
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            print(f'  WARNING: subtitle file not found: {full_path}')
            return ''
    try:
        for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                with open(full_path, encoding=enc) as f:
                    raw = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f'  WARNING: could not decode {full_path}')
            return ''
    except Exception as e:
        print(f'  WARNING: error reading {full_path}: {e}')
        return ''

    blocks = re.split(r'\r?\n\r?\n', raw.strip())
    dialogue_lines = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines and lines[0].isdigit():
            lines = lines[1:]
        if lines and '-->' in lines[0]:
            lines = lines[1:]
        if not lines:
            continue
        text = ' '.join(lines)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\{[^}]+\}', '', text)
        dialogue_lines.append(text)
    full = '\n'.join(dialogue_lines)
    return full[:max_chars]

def parse_options(opts_field):
    try:
        return ast.literal_eval(opts_field)
    except:
        return eval(opts_field)

def build_subtitle_prompt(question, item, dialogue_text):
    label_chars = 'ABCDE'
    opts = parse_options(item['options'])
    options_text = ''
    for i, opt in enumerate(opts):
        if i >= len(label_chars):
            break
        text = ', '.join(opt) if isinstance(opt, list) else str(opt)
        options_text += f'\n{label_chars[i]}. {text}'

    if dialogue_text.strip():
        return (
            f"Watch the full video carefully and use the dialogue subtitles "
            f"below to help track each character's appearance.\n\n"
            f"Episode dialogue:\n{dialogue_text}\n\n"
            f"Question: {question}\n"
            f"Options:{options_text}\n\n"
            f"Reply with ONLY a single letter: A, B, C, D, or E."
        )
    else:
        return (
            f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {question}\n"
            f"Options:{options_text}\n\n"
            f"Reply with ONLY a single letter: A, B, C, D, or E."
        )

def load_frames(vpath, n=16):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr)-1, n, dtype=int)
    frames = vr.get_batch(idx).numpy()
    return [Image.fromarray(frames[i]) for i in range(n)]

# ========================================================================
# Verify subtitle loading on first 3 BBT items
# ========================================================================
print('Verifying subtitle loader on first 3 questions...')
for item in bbt_val[:3]:
    qid = str(item['question_id'])
    srt_path = item.get('video_subtitles', '')
    dialogue = load_srt_dialogue(srt_path)
    preview = dialogue[:200].replace('\n', ' / ')
    print(f'  Q{qid}: srt_path={srt_path}')
    print(f'    Dialogue length: {len(dialogue)} chars')
    print(f'    First 200 chars: {preview}...')
print()

test_dialogue = load_srt_dialogue(bbt_val[0].get('video_subtitles', ''))
if len(test_dialogue) < 100:
    print('*** Subtitle loading appears broken — halt and investigate. ***')
    sys.exit(1)

# ========================================================================
# Generic runner
# ========================================================================
def run_subtitle_experiment(model_name, model_runner, out_file):
    print(f'\n{"="*60}')
    print(f'  Subtitle ablation: {model_name}')
    print('=' * 60)
    done, results = set(), []
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
        done = {str(r['question_id']) for r in results}
    remaining = [d for d in bbt_val if str(d['question_id']) not in done]
    print(f'  Already done: {len(done)}, Remaining: {len(remaining)}')

    for item in remaining:
        qid    = str(item['question_id'])
        vpath  = os.path.join(BASE_DIR, item['video_path_mp4'])
        srt_path = item.get('video_subtitles', '')
        dialogue = load_srt_dialogue(srt_path)
        prompt = build_subtitle_prompt(item['question'], item, dialogue)
        try:
            frames = load_frames(vpath, 16)
            resp = model_runner(frames, prompt)
            pred = next((c for c in resp.upper() if c in 'ABCDE'), 'X')
        except Exception as e:
            print(f'  ERROR Q{qid}: {e}')
            pred = 'X'
        correct = pred == item['answer']
        results.append({
            'question_id':    qid,
            'correct_label':  item['answer'],
            'prediction':     pred,
            'correct':        correct,
            'subtitle_chars': len(dialogue),
            'srt_path':       srt_path,
        })
        flag = 'OK' if correct else '--'
        print(f'  [{flag}] Q{qid} | GT={item["answer"]} Pred={pred} (subs={len(dialogue)} chars)')
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        torch.cuda.empty_cache()

    n_total = len(results); n_correct = sum(r['correct'] for r in results)
    print(f'\n  {model_name} RESULT: {n_correct}/{n_total} = {n_correct/max(1,n_total)*100:.1f}%')
    return results

# ========================================================================
# InternVL2-8B
# ========================================================================
print('Loading InternVL2-8B...')
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

transform = T.Compose([
    T.Resize((448,448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
tokenizer = AutoTokenizer.from_pretrained(
    INTERNVL_PATH, trust_remote_code=True, use_fast=False)
internvl = AutoModel.from_pretrained(
    INTERNVL_PATH, torch_dtype=torch.float16,
    device_map='cuda:0', trust_remote_code=True)
internvl.eval()
print('InternVL2 loaded.\n')

def internvl_run(frames, prompt_text):
    pv = torch.stack([transform(f) for f in frames]).half().cuda()
    resp = internvl.chat(
        tokenizer, pv,
        f'Frame1: <image>\n{prompt_text}',
        dict(max_new_tokens=10, do_sample=False))
    return resp.strip()

run_subtitle_experiment('InternVL2-8B', internvl_run, OUT_INTERNVL)
del internvl, tokenizer
torch.cuda.empty_cache()

# ========================================================================
# Qwen2.5-VL-7B
# ========================================================================
print('\nLoading Qwen2.5-VL-7B...')
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_PATH, torch_dtype=torch.float16,
    device_map='cuda:0', trust_remote_code=True)
qwen.eval()
print('Qwen2.5-VL loaded.\n')

def qwen_run(frames, prompt_text):
    messages = [{
        'role': 'user',
        'content': (
            [{'type':'image','image':f} for f in frames] +
            [{'type':'text','text':prompt_text}]
        )
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt'
    ).to('cuda')
    with torch.no_grad():
        out = qwen.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = out[0][inputs.input_ids.shape[1]:]
    return qwen_processor.decode(generated, skip_special_tokens=True).strip()

run_subtitle_experiment('Qwen2.5-VL-7B', qwen_run, OUT_QWEN)
del qwen, qwen_processor
torch.cuda.empty_cache()

# ========================================================================
# LLaVA-NeXT-Video-7B
# ========================================================================
print('\nLoading LLaVA-NeXT-Video-7B...')
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

llava_processor = LlavaNextVideoProcessor.from_pretrained(LLAVA_PATH)
llava = LlavaNextVideoForConditionalGeneration.from_pretrained(
    LLAVA_PATH, torch_dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True
)
llava.eval()
print('LLaVA-NeXT-Video loaded.\n')

def llava_run(frames, prompt_text):
    np_frames = np.stack([np.array(f) for f in frames])
    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'video'},
            {'type': 'text', 'text': prompt_text},
        ],
    }]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(
        text=[prompt], videos=[np_frames],
        padding=True, return_tensors='pt'
    ).to('cuda', torch.float16)
    with torch.no_grad():
        out = llava.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = out[0][inputs.input_ids.shape[1]:]
    return llava_processor.decode(generated, skip_special_tokens=True).strip()

run_subtitle_experiment('LLaVA-NeXT-Video-7B', llava_run, OUT_LLAVA)
del llava, llava_processor
torch.cuda.empty_cache()

# ========================================================================
# McNemar comparison vs baseline full-episode results
# ========================================================================
print('\n' + '='*60)
print('  COMPARISON vs baseline (no subtitles)')
print('='*60)

for name, sub_file, base_file in [
    ('InternVL2-8B',        OUT_INTERNVL, OUT_DIR_INTERNVL / 'predictions.json'),
    ('Qwen2.5-VL-7B',       OUT_QWEN,     OUT_DIR_QWEN     / 'predictions.json'),
    ('LLaVA-NeXT-Video-7B', OUT_LLAVA,    OUT_DIR_LLAVA    / 'predictions.json'),
]:
    if not sub_file.exists() or not base_file.exists():
        print(f'\n  {name}: missing one of the files, skipping comparison')
        continue
    with open(sub_file) as f:
        sub = json.load(f)
    with open(base_file) as f:
        base = json.load(f)
    sub_map  = {str(r['question_id']): r for r in sub}
    base_map = {str(r['question_id']): r for r in base}
    common_qids = sorted(set(sub_map) & set(base_map))
    n = len(common_qids)
    if n == 0:
        print(f'\n  {name}: no overlapping question_ids')
        continue
    n_sub_correct  = sum(sub_map[q]['correct']  for q in common_qids)
    n_base_correct = sum(base_map[q]['correct'] for q in common_qids)
    b = sum(1 for q in common_qids
            if base_map[q]['correct'] and not sub_map[q]['correct'])
    c = sum(1 for q in common_qids
            if sub_map[q]['correct'] and not base_map[q]['correct'])

    print(f'\n  {name} (n={n} common questions):')
    print(f'    Baseline accuracy:  {n_base_correct}/{n} = {n_base_correct/n*100:.1f}%')
    print(f'    Subtitle accuracy:  {n_sub_correct}/{n} = {n_sub_correct/n*100:.1f}%')
    print(f'    Delta:              {(n_sub_correct - n_base_correct)/n*100:+.1f}pp')
    print(f'    McNemar (b={b}, c={c}):')

    if (b + c) > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2 as chi2_dist
            p = 1 - chi2_dist.cdf(chi2, df=1)
            print(f'      chi2={chi2:.2f}, p={p:.4f}')
        except ImportError:
            print(f'      chi2={chi2:.2f} (install scipy for p-value)')
    else:
        print('      No discordant pairs.')

print('\n' + '='*60)
print('  DONE')
print('='*60)
