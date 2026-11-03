"""Friends/Castle name swap for InternVL2, Qwen and LLaVA (124 pairs per show).

Keeps the 40 pairs per show of the pilot run and adds pairs sampled with seed 42, so all
three models are asked identical (question, original name, replacement name) triples.
Models are loaded one at a time. Resumable.
"""
import os, json, ast, random, sys, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')

INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH     = r'D:\models\qwen25_vl_7b'
LLAVA_PATH    = r'D:\models\llava_next_video_7b'

BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
TR_ANN   = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'

OUT_INTERNVL = Path(r'D:\eval_workspace\results\global_appearance\InternVL2-8B')
OUT_QWEN     = Path(r'D:\eval_workspace\results\global_appearance\Qwen25VL')
OUT_LLAVA    = Path(r'D:\eval_workspace\results\global_appearance\LLaVA-NeXT-Video-7B')
for d in (OUT_INTERNVL, OUT_QWEN, OUT_LLAVA):
    d.mkdir(parents=True, exist_ok=True)

FRIENDS_CHARS = ['Ross', 'Rachel', 'Monica', 'Chandler', 'Joey', 'Phoebe']
CASTLE_CHARS  = ['Castle', 'Beckett', 'Esposito', 'Ryan', 'Lanie', 'Alexis',
                 'Martha', 'Gates', 'Geller']

TARGET_N = 124   # match BBT scope
random.seed(42)

# ---- Load annotations ----
with open(TR_ANN) as f:
    tr_data = [d for d in json.load(f) if d.get('is_visual', False)]
all_by_id = {str(d['question_id']): d for d in tr_data}

friends_items = [d for d in tr_data if 'friends' in d['video_path_mp4'].lower()]
castle_items  = [d for d in tr_data if 'castle'  in d['video_path_mp4'].lower()]
print(f'Friends total: {len(friends_items)}, Castle total: {len(castle_items)}')

# ---- Sample/extend swap pair lists, REUSING existing Qwen pairs ----
def load_existing_pairs(qwen_file):
    if not qwen_file.exists():
        return []
    with open(qwen_file) as f:
        data = json.load(f)
    return [{
        'qid': str(r['question_id']),
        'original': r.get('original_name') or r.get('original_char'),
        'swap':     r.get('swap_name')     or r.get('swap_char'),
    } for r in data]

def build_swap_list(items, characters, existing_pairs, target_n):
    """Reuse existing pairs, add new ones up to target_n."""
    out = []
    seen_qids = set()
    # Keep existing pairs first
    for p in existing_pairs:
        if p['qid'] in all_by_id and p['original'] and p['swap']:
            out.append(p)
            seen_qids.add(p['qid'])

    # Sample new questions
    candidates = []
    for item in items:
        qid = str(item['question_id'])
        if qid in seen_qids:
            continue
        named_char = next((c for c in characters if c in item['question']), None)
        if named_char is None:
            continue
        others = [c for c in characters
                  if c != named_char and c not in item['question']]
        if not others:
            continue
        swap_target = random.choice(others)
        candidates.append({'qid': qid, 'original': named_char, 'swap': swap_target})

    random.shuffle(candidates)
    needed = max(0, target_n - len(out))
    out.extend(candidates[:needed])
    return out

friends_pairs = build_swap_list(
    friends_items, FRIENDS_CHARS,
    load_existing_pairs(OUT_QWEN / 'predictions_name_swap_friends.json'),
    TARGET_N,
)
castle_pairs = build_swap_list(
    castle_items, CASTLE_CHARS,
    load_existing_pairs(OUT_QWEN / 'predictions_name_swap_castle.json'),
    TARGET_N,
)
print(f'Friends pairs: {len(friends_pairs)} (target {TARGET_N})')
print(f'Castle pairs:  {len(castle_pairs)} (target {TARGET_N})')

# Persist the canonical pair lists so all three models use them
PAIRS_DIR = Path(r'D:\eval_workspace\results\global_appearance')
with open(PAIRS_DIR / 'expanded_pairs_friends.json', 'w') as f:
    json.dump(friends_pairs, f, indent=2)
with open(PAIRS_DIR / 'expanded_pairs_castle.json', 'w') as f:
    json.dump(castle_pairs, f, indent=2)

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

def load_frames(vpath, n=16):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr)-1, n, dtype=int)
    frames = vr.get_batch(idx).numpy()
    return [Image.fromarray(frames[i]) for i in range(n)]

# ---- Generic runner ----
def run_swap_set(model_name, run_fn, pairs, out_file, show_name):
    done, results = set(), []
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
        done = {str(r['question_id']) for r in results}
    remaining = [p for p in pairs if p['qid'] not in done]
    print(f'  [{model_name} {show_name}] done={len(done)} remaining={len(remaining)}')

    for entry in remaining:
        qid  = entry['qid']
        item = all_by_id.get(qid)
        if item is None:
            continue
        vpath = os.path.join(BASE_DIR, item['video_path_mp4'])
        orig_prompt = build_mc_prompt(item['question'], item)
        swap_q = item['question'].replace(entry['original'], entry['swap'])
        swap_prompt = build_mc_prompt(swap_q, item)

        try:
            frames = load_frames(vpath, 16)
            orig_resp = run_fn(frames, orig_prompt)
            orig_pred = next((c for c in orig_resp.upper() if c in 'ABCDE'), 'X')
            swap_resp = run_fn(frames, swap_prompt)
            swap_pred = next((c for c in swap_resp.upper() if c in 'ABCDE'), 'X')
        except Exception as e:
            print(f'    ERROR Q{qid}: {e}')
            continue

        answer_changed = orig_pred != swap_pred
        results.append({
            'question_id':    qid,
            'original_name':  entry['original'],
            'swap_name':      entry['swap'],
            'correct_label':  item['answer'],
            'orig_pred':      orig_pred,
            'swap_pred':      swap_pred,
            'orig_correct':   orig_pred == item['answer'],
            'answer_changed': answer_changed,
        })
        print(f'    Q{qid} {entry["original"]}->{entry["swap"]} '
              f'orig={orig_pred} swap={swap_pred} changed={"Y" if answer_changed else "N"}')
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        torch.cuda.empty_cache()

    n = len(results)
    changed = sum(r['answer_changed'] for r in results)
    print(f'  {model_name} {show_name}: {changed}/{n} = {changed/max(1,n)*100:.1f}% sensitivity\n')

# ========================================================================
# Model loaders
# ========================================================================
def with_internvl(fn):
    print('\nLoading InternVL2-8B...')
    from transformers import AutoTokenizer, AutoModel
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Resize((448,448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    tok = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)
    m = AutoModel.from_pretrained(INTERNVL_PATH, torch_dtype=torch.float16,
                                  device_map='cuda:0', trust_remote_code=True)
    m.eval()
    print('InternVL2 loaded.')
    def run(frames, prompt):
        pv = torch.stack([transform(f) for f in frames]).half().cuda()
        return m.chat(tok, pv, f'Frame1: <image>\n{prompt}',
                      dict(max_new_tokens=10, do_sample=False)).strip()
    fn(run, 'InternVL2-8B', OUT_INTERNVL)
    del m, tok; torch.cuda.empty_cache()

def with_qwen(fn):
    print('\nLoading Qwen2.5-VL-7B...')
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16,
        device_map='cuda:0', trust_remote_code=True)
    m.eval()
    print('Qwen2.5-VL loaded.')
    def run(frames, prompt):
        messages = [{'role':'user','content':(
            [{'type':'image','image':f} for f in frames] +
            [{'type':'text','text':prompt}]
        )}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(messages)
        inputs = proc(text=[text], images=ii, videos=vi,
                      padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return proc.decode(gen, skip_special_tokens=True).strip()
    fn(run, 'Qwen2.5-VL-7B', OUT_QWEN)
    del m, proc; torch.cuda.empty_cache()

def with_llava(fn):
    print('\nLoading LLaVA-NeXT-Video-7B...')
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
    proc = LlavaNextVideoProcessor.from_pretrained(LLAVA_PATH)
    m = LlavaNextVideoForConditionalGeneration.from_pretrained(
        LLAVA_PATH, torch_dtype=torch.float16,
        device_map='cuda:0', low_cpu_mem_usage=True)
    m.eval()
    print('LLaVA-NeXT-Video loaded.')
    def run(frames, prompt):
        np_frames = np.stack([np.array(f) for f in frames])
        conv = [{'role':'user','content':[
            {'type':'video'},{'type':'text','text':prompt}
        ]}]
        p = proc.apply_chat_template(conv, add_generation_prompt=True)
        inputs = proc(text=[p], videos=[np_frames],
                      padding=True, return_tensors='pt').to('cuda', torch.float16)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return proc.decode(gen, skip_special_tokens=True).strip()
    fn(run, 'LLaVA-NeXT-Video-7B', OUT_LLAVA)
    del m, proc; torch.cuda.empty_cache()

# ========================================================================
# Run all three models on expanded Friends + Castle pairs
# ========================================================================
def runner_for(run_fn, model_name, out_dir):
    run_swap_set(model_name, run_fn, friends_pairs,
                 out_dir / 'predictions_name_swap_friends_expanded.json',
                 'Friends')
    run_swap_set(model_name, run_fn, castle_pairs,
                 out_dir / 'predictions_name_swap_castle_expanded.json',
                 'Castle')

print('=' * 70)
print('  EXPANDED NAME-SWAP EVALUATION')
print('  Friends and Castle to n~=124 across InternVL2, Qwen, LLaVA')
print('=' * 70)

start = time.time()
try:
    with_internvl(runner_for)
    with_qwen(runner_for)
    with_llava(runner_for)
except KeyboardInterrupt:
    print('\n*** Interrupted. Re-run to resume. ***')
    sys.exit(0)

elapsed = (time.time() - start) / 3600
print(f'\n{"="*70}')
print(f'  ALL DONE in {elapsed:.2f} hours')
print('=' * 70)
