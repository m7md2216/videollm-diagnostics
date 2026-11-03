"""Robustness runs for the valid-option swap: documented InternVL2 input, frames from both
characters' scenes, labels 1-5, reversed option order (resumable; nothing existing is overwritten).

The valid-option pairs are rebuilt EXACTLY from the existing result files
(same questions, same option sets, same correct letters, same swapped wording):
  BBT:     <model>/predictions_valid_option_swap.json      (+ valid_option_candidates.json)
  Friends: <model>/predictions_valid_option_swap_friends.json
  Castle:  <model>/predictions_valid_option_swap_castle.json

EXP 1  InternVL2 with its documented video input (one <image> slot per frame,
       num_patches_list), instead of all 16 frames in the first slot:
         - full-episode accuracy on BBT (129 questions)
         - valid-option swap, 3 shows, video + black frames
EXP 2  "Pair oracle": valid-option swap where the 16 frames come from the annotated outfit
       windows of BOTH characters (padded with uniform frames to 16). Same frames for both
       questions. InternVL2 (documented format) and Qwen. Video only.
EXP 3  Answer-format robustness of the valid-option swap, uniform frames, video only:
         - labels 1-5 instead of A-E (same positions)
         - option order reversed (A-E labels), so every option changes position
       InternVL2 (documented format) and Qwen.

Run order: EXP1 -> EXP2 -> EXP3 for InternVL2, then EXP2 -> EXP3 for Qwen.
Outputs: results/global_appearance/<model>/r2_<name>.json
    python run_robustness.py --dry      # check rebuilt items, no model loaded
    python run_robustness.py            # everything
"""
import os, sys, json, ast, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import decord
decord.bridge.set_bridge('torch')

VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH = r'D:\models\qwen25_vl_7b'
RES = Path(r'D:\eval_workspace\results\global_appearance')
N = 16
AE, NUM = list('ABCDE'), list('12345')

val = json.load(open(VAL_ANN, encoding='utf-8'))
tr = json.load(open(TR_ANN, encoding='utf-8'))
ANN = {'bbt': {str(d['question_id']): d for d in val}, 'friends': {}, 'castle': {}}
for d in tr:
    p = d.get('video_path_mp4', '').lower().replace('\\', '/')
    for s in ('friends', 'castle'):
        if f'/{s}/' in p or p.startswith(f'{s}/'):
            ANN[s][str(d['question_id'])] = d


def opts_of(a):
    try:
        return ast.literal_eval(a['options'])
    except Exception:
        return eval(a['options'])


def gold_of(a):
    return opts_of(a)[int(a['answer_idx'])]


def norm(o):
    return tuple(str(x).strip().lower() for x in (o if isinstance(o, list) else [o]))


def opt_text(o):
    return ', '.join(o) if isinstance(o, list) else str(o)


def secs(t):
    h, m, s = str(t).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def window_mids(tg):
    mids = []
    for g in ast.literal_eval(tg):
        if isinstance(g, list) and g and isinstance(g[0], list):
            mids += [(secs(w[0]) + secs(w[1])) / 2 for w in g]
        elif isinstance(g, list) and g and isinstance(g[0], str):
            mids.append((secs(g[0]) + secs(g[1])) / 2)
    return mids


# ---------------------------------------------------------------- rebuild the 266 pairs
cand_bbt = {c['qid']: c for c in
            json.load(open(RES / 'valid_option_candidates.json', encoding='utf-8'))['existing_pairs']['InternVL2-8B']}
SRC = {'bbt': 'predictions_valid_option_swap.json', 'friends': 'predictions_valid_option_swap_friends.json',
       'castle': 'predictions_valid_option_swap_castle.json'}
items, problems = [], []
for show, fn in SRC.items():
    rows = [r for r in json.load(open(RES / 'Qwen25VL' / fn, encoding='utf-8')) if r['mode'] == 'video']
    for r in rows:
        q = str(r['question_id']); a = ANN[show].get(q)
        if a is None:
            problems.append((show, q, 'no annotation')); continue
        if show == 'bbt':
            ysrc = str(cand_bbt[q]['target_qid']); y_gold = cand_bbt[q]['y_gold']
            ya = ANN['bbt'].get(ysrc) or next((d for d in tr if str(d['question_id']) == ysrc), None)
        else:
            ysrc = str(r['y_source_qid']); ya = ANN[show].get(ysrc); y_gold = gold_of(ya)
        opts = opts_of(a)[:5]
        xi, yi = AE.index(r['x_label']), AE.index(r['y_label'])
        if not r.get('y_already_present'):
            opts[yi] = y_gold
        if norm(opts[xi]) != norm(gold_of(a)) or norm(opts[yi]) != norm(y_gold):
            problems.append((show, q, 'option mismatch')); continue
        items.append(dict(show=show, qid=q, x=r['original_char'], y=r['swap_char'], video=a['video_path_mp4'],
                          q_x=a['question'], q_y=r['question_y'], opts=opts, xi=xi, yi=yi,
                          mids=sorted(window_mids(a['temporal_grounding']) +
                                      (window_mids(ya['temporal_grounding']) if ya else []))))
print(f'rebuilt pairs: {len(items)}  ' + str({s: sum(i["show"] == s for i in items) for s in SRC}))
print(f'problems: {len(problems)} {problems[:5]}')
bbt_acc = [d for d in val if d.get('is_visual', False) and 'bbt' in d['video_path_mp4'].lower()]
print(f'BBT accuracy questions: {len(bbt_acc)}')


def prompt(question, opts, labels):
    body = ''.join(f'\n{labels[i]}. {opt_text(o)}' for i, o in enumerate(opts))
    return (f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {question}\nOptions:{body}\n\n"
            f"Reply with ONLY a single letter: {', '.join(labels[:len(opts)])}.")


def mc_prompt(question, opts):          # identical to the original full-episode prompt
    body = ''.join(f'\n{AE[i]}. {opt_text(o)}' for i, o in enumerate(opts[:5]))
    return (f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {question}\nOptions:{body}\n\n"
            f"Reply with ONLY a single letter: A, B, C, D, or E.")


ex = items[0]
print('\nExample rebuilt pair:', ex['show'], ex['qid'], ex['x'], '->', ex['y'])
print(prompt(ex['q_y'], ex['opts'], AE)); print('X correct', AE[ex['xi']], 'Y correct', AE[ex['yi']])
print('pair-oracle window midpoints (s):', [round(m) for m in ex['mids']])
if '--dry' in sys.argv:
    sys.exit('\nDry run only.')

# ---------------------------------------------------------------- frames
_vr = {}


def vr_of(path):
    if path not in _vr:
        _vr.clear()
        _vr[path] = decord.VideoReader(os.path.join(BASE_DIR, path))
    return _vr[path]


def uniform_idx(n_frames, n=N):
    return list(np.linspace(0, n_frames - 1, n, dtype=int))


def frames_uniform(path):
    vr = vr_of(path)
    arr = vr.get_batch(uniform_idx(len(vr))).numpy()
    return [Image.fromarray(a) for a in arr]


def frames_pair_oracle(path, mids):
    """One frame per annotated window of both characters (up to 16, evenly chosen if more),
    padded with uniform frames to 16, in time order."""
    vr = vr_of(path); fps = vr.get_avg_fps(); n = len(vr)
    idx = sorted({min(n - 1, max(0, int(t * fps))) for t in mids})
    if len(idx) > N:
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, N, dtype=int)]
    k = len(idx)
    for u in uniform_idx(n):
        if len(idx) >= N:
            break
        if u not in idx:
            idx.append(u)
    idx = sorted(idx)
    arr = vr.get_batch(idx).numpy()
    return [Image.fromarray(a) for a in arr], k


BLACK = [Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)) for _ in range(N)]


def first_label(resp, labels):
    return next((ch for ch in resp.strip().upper() if ch in labels), 'X')


# ---------------------------------------------------------------- generic runners
def save(path, rows):
    json.dump(rows, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def load(path):
    return json.load(open(path, encoding='utf-8')) if path.exists() else []


def run_valid_option(folder, infer, name, frame_mode, variant='AE', modes=('video',)):
    out = RES / folder / f'r2_{name}.json'
    rows = load(out); done = {(r['show'], r['qid'], r['mode']) for r in rows}
    for mode in modes:
        todo = [it for it in items if (it['show'], it['qid'], mode) not in done]
        print(f'\n--- {folder} / {name} / {mode}: {len(items) - len(todo)} done, {len(todo)} to go')
        for it in todo:
            opts, xi, yi, labels = list(it['opts']), it['xi'], it['yi'], AE
            if variant == 'NUM':
                labels = NUM
            elif variant == 'REV':
                opts = opts[::-1]; xi, yi = 4 - xi, 4 - yi
            try:
                n_or = None
                if mode == 'black':
                    fr = BLACK
                elif frame_mode == 'pair_oracle':
                    fr, n_or = frames_pair_oracle(it['video'], it['mids'])
                else:
                    fr = frames_uniform(it['video'])
                rx = infer(fr, prompt(it['q_x'], opts, labels))
                ry = infer(fr, prompt(it['q_y'], opts, labels))
            except Exception as e:
                print(f"  ERROR {it['show']} Q{it['qid']}: {e}"); continue
            px, py = first_label(rx, labels), first_label(ry, labels)
            # store predictions mapped back to ORIGINAL A-E positions, so analyses are comparable
            def back(p):
                if p == 'X':
                    return 'X'
                i = labels.index(p)
                return AE[4 - i] if variant == 'REV' else AE[i]
            rows.append(dict(show=it['show'], qid=it['qid'], mode=mode, variant=variant, frames=frame_mode,
                             original_char=it['x'], swap_char=it['y'],
                             x_label=AE[it['xi']], y_label=AE[it['yi']], pred_x=back(px), pred_y=back(py),
                             raw_x=rx, raw_y=ry, n_oracle_frames=n_or))
            save(out, rows)
            print(f"  {it['show']:7s} Q{it['qid']} {mode} {it['x']}->{it['y']}  X={AE[it['xi']]} Y={AE[it['yi']]}"
                  f" | asked X {back(px)}  asked Y {back(py)}")
            torch.cuda.empty_cache()


def run_accuracy(folder, infer, name):
    out = RES / folder / f'r2_{name}.json'
    rows = load(out); done = {r['qid'] for r in rows}
    todo = [d for d in bbt_acc if str(d['question_id']) not in done]
    print(f'\n--- {folder} / {name}: {len(bbt_acc) - len(todo)} done, {len(todo)} to go')
    for d in todo:
        try:
            r = infer(frames_uniform(d['video_path_mp4']), mc_prompt(d['question'], opts_of(d)))
        except Exception as e:
            print(f"  ERROR Q{d['question_id']}: {e}"); continue
        p = first_label(r, AE)
        rows.append(dict(qid=str(d['question_id']), correct_label=d['answer'], prediction=p,
                         correct=p == d['answer'], raw=r))
        save(out, rows)
        torch.cuda.empty_cache()
    print(f'  accuracy: {sum(r["correct"] for r in rows)}/{len(rows)}')


# ---------------------------------------------------------------- InternVL2 (documented video format)
print('=' * 60); print('InternVL2-8B, documented video input'); print('=' * 60)
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
tok = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)
mdl = AutoModel.from_pretrained(INTERNVL_PATH, torch_dtype=torch.float16, device_map='cuda:0',
                                trust_remote_code=True).eval()


def iv_video(frames, text):
    # As in the InternVL2 model card's video example with max_num=1: one 448px tile per frame,
    # one <image> placeholder per frame, and num_patches_list telling chat() which is which.
    pv = torch.stack([tf(f.convert('RGB')) for f in frames]).half().cuda()
    prefix = ''.join(f'Frame{i + 1}: <image>\n' for i in range(len(frames)))
    return mdl.chat(tok, pv, prefix + text, dict(max_new_tokens=10, do_sample=False),
                    num_patches_list=[1] * len(frames))


run_accuracy('InternVL2-8B', iv_video, 'ivl_videofmt_accuracy_bbt')
run_valid_option('InternVL2-8B', iv_video, 'ivl_videofmt_validopt', 'uniform', 'AE', ('video', 'black'))
run_valid_option('InternVL2-8B', iv_video, 'pair_oracle_validopt', 'pair_oracle')
run_valid_option('InternVL2-8B', iv_video, 'labels_num_validopt', 'uniform', 'NUM')
run_valid_option('InternVL2-8B', iv_video, 'labels_rev_validopt', 'uniform', 'REV')
del mdl, tok; torch.cuda.empty_cache()

# ---------------------------------------------------------------- Qwen (unchanged inference code)
print('=' * 60); print('Qwen2.5-VL-7B'); print('=' * 60)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
proc = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.float16,
                                                        device_map='cuda:0', trust_remote_code=True).eval()


def qw(frames, text):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': f} for f in frames] +
                 [{'type': 'text', 'text': text}]}]
    t = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(messages)
    inputs = proc(text=[t], images=ii, videos=vi, padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = qm.generate(**inputs, max_new_tokens=10, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


run_valid_option('Qwen25VL', qw, 'pair_oracle_validopt', 'pair_oracle')
run_valid_option('Qwen25VL', qw, 'labels_num_validopt', 'uniform', 'NUM')
run_valid_option('Qwen25VL', qw, 'labels_rev_validopt', 'uniform', 'REV')
print('\nALL DONE. Now run: python analyze_robustness.py')
