"""Valid-option name swap on FRIENDS and CASTLE (same design as the BBT run).

Pairs: the X -> Y name swaps already used on Friends/Castle
       (Qwen25VL/predictions_name_swap_<show>_expanded.json; all 3 models used the same pairs).
Kept : X and Y are main/recurring cast, and Y has their own global-appearance question in
       the same episode, so Y's correct outfit sequence is known.
       ("Geller", a Friends surname that appeared in the Castle name list, is skipped.)

Exactly as on BBT:
  ONE modified option set per pair: original options A-E, one distractor (not X's gold,
  not "I don't know") replaced by Y's correct sequence, chosen with seed 2026 per question.
  Asked about X -> X's option is correct; asked about Y -> Y's option is correct.
  Same frames, same options; only the name differs.
  Modes: video (16 uniform frames) and black (16 black 448x448 frames).
  Same prompt, same InternVL2 / Qwen inference code, greedy, 10 new tokens.

Output (new files, resumable; nothing existing is touched):
  results/global_appearance/<model>/predictions_valid_option_swap_<show>.json
"""
import os, json, ast, random, re, sys, torch, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import decord
decord.bridge.set_bridge('torch')

TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH = r'D:\models\qwen25_vl_7b'
RES = Path(r'D:\eval_workspace\results\global_appearance')

SHOWS = ['friends', 'castle']
MODES = ['video', 'black']
SEED = 2026
N_FRAMES = 16
LETTERS = list('ABCDE')
CAST = {'friends': ['rachel', 'monica', 'phoebe', 'ross', 'chandler', 'joey'],
        'castle': ['castle', 'beckett', 'alexis', 'martha', 'lanie', 'gates', 'esposito', 'ryan']}
FEMALE = {'rachel', 'monica', 'phoebe', 'beckett', 'alexis', 'martha', 'lanie', 'gates'}


def get_options(item):
    try:
        return ast.literal_eval(item['options'])
    except Exception:
        return None


def opt_text(opt):
    return ', '.join(opt) if isinstance(opt, list) else str(opt)


def norm(opt):
    return tuple(str(x).strip().lower() for x in (opt if isinstance(opt, list) else [opt]))


def is_idk(opt):
    return 'know' in opt_text(opt).lower()


# Full-name forms, so "Dr. Ross Geller" -> "Phoebe" (not "Dr. Phoebe Geller")
FIRST = {'castle': ['Richard', 'Rick'], 'beckett': ['Kate', 'Katherine', 'Katie'],
         'esposito': ['Javier'], 'ryan': ['Kevin'], 'gates': ['Victoria']}
LAST = {'ross': ['Geller'], 'monica': ['Geller'], 'rachel': ['Green'], 'chandler': ['Bing'],
        'joey': ['Tribbiani'], 'phoebe': ['Buffay'], 'lanie': ['Parish'], 'alexis': ['Castle'],
        'martha': ['Rodgers']}
TITLE = r"(?:(?:Dr|Mr|Mrs|Ms|Miss|Detective|Det|Captain|Capt|Officer|Lieutenant|Lt|Sergeant|Sgt)\.?\s+)?"


def name_pattern(tok):
    first = FIRST.get(tok, [])
    last = LAST.get(tok, [])
    pat = TITLE
    if first:
        pat += r'(?:(?:' + '|'.join(first) + r')\s+)?'
    pat += r'\b' + tok.title() + r'\b'
    if last:
        pat += r'(?:\s+(?:' + '|'.join(last) + r')\b)?'
    return pat


def swap_name(q, x, y):
    return re.sub(name_pattern(x), y.title(), q)


def build_prompt(question, opts):   # identical to the BBT valid-option run
    options_text = ''.join(f'\n{LETTERS[i]}. {opt_text(o)}' for i, o in enumerate(opts))
    return (f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {question}\n"
            f"Options:{options_text}\n\n"
            f"Reply with ONLY a single letter: {', '.join(LETTERS[:len(opts)])}.")


# ---------------------------------------------------------------- annotations
ann_all = json.load(open(TR_ANN, encoding='utf-8'))
ann, by_ep_char = {}, defaultdict(list)
for a in ann_all:
    path = a.get('video_path_mp4', '').lower().replace('\\', '/')
    show = next((s for s in SHOWS if f'/{s}/' in path or path.startswith(f'{s}/')), None)
    if show is None:
        continue
    ann[str(a['question_id'])] = a
    hits = [c for c in CAST[show] if re.search(rf'\b{c}\b', a.get('question', '').lower())]
    opts = get_options(a)
    try:
        gold = opts[int(a['answer_idx'])]
    except Exception:
        gold = None
    if len(hits) == 1 and gold is not None and not is_idk(gold):
        by_ep_char[(path, hits[0])].append((str(a['question_id']), gold))

# ---------------------------------------------------------------- build items
items = {s: [] for s in SHOWS}
for show in SHOWS:
    pf = RES / 'Qwen25VL' / f'predictions_name_swap_{show}_expanded.json'
    pairs = json.load(open(pf, encoding='utf-8'))
    skip = Counter()
    for r in pairs:
        qid = str(r['question_id'])
        xr = str(r.get('original_name') or r.get('original_char') or '')
        yr = str(r.get('swap_name') or r.get('swap_char') or '')
        x, y = xr.lower(), yr.lower()
        if x not in CAST[show] or y not in CAST[show]:
            skip[f'name not in cast ({xr}->{yr})'] += 1; continue
        a = ann.get(qid)
        if a is None:
            skip['no annotation'] += 1; continue
        opts = get_options(a)
        gx = int(a['answer_idx'])
        if opts is None or gx > 4 or len(opts) < 5:
            skip['gold outside A-E'] += 1; continue
        opts = opts[:5]
        path = a['video_path_mp4'].lower().replace('\\', '/')
        ys = by_ep_char.get((path, y), [])
        if not ys:
            skip['Y has no question in episode'] += 1; continue
        y_qid, y_gold = ys[0]
        if norm(y_gold) == norm(opts[gx]):
            skip['Y sequence = X sequence'] += 1; continue
        existing = [i for i in range(5) if norm(opts[i]) == norm(y_gold)]
        new_opts = list(opts)
        if existing:                       # Y's answer is already a distractor: use it as is
            ry = existing[0]
        else:
            distractors = [i for i in range(5) if i != gx and not is_idk(opts[i])]
            if not distractors:
                skip['no replaceable distractor'] += 1; continue
            ry = random.Random(f'{SEED}-{qid}').choice(distractors)
            new_opts[ry] = y_gold
        q_x = a['question']
        named = [c for c in CAST[show] if re.search(rf'\b{c}\b', q_x.lower())]
        if named != [x]:
            skip['question names a different or second cast member'] += 1; continue
        q_y = swap_name(q_x, x, y)
        if q_y == q_x:
            skip['name not found in question text'] += 1; continue
        nx = len(opts[gx]) if isinstance(opts[gx], list) else 1
        ny = len(y_gold) if isinstance(y_gold, list) else 1
        items[show].append(dict(
            qid=qid, show=show, x=x, y=y, y_qid=y_qid, video=a['video_path_mp4'],
            q_x=q_x, q_y=q_y, opts=new_opts, x_idx=gx, y_idx=ry,
            y_already_present=bool(existing), len_match=(nx == ny),
            gender='same' if (x in FEMALE) == (y in FEMALE) else 'cross'))
    g = Counter(it['gender'] for it in items[show])
    print(f'\n{show.upper()}: {len(pairs)} pairs -> usable {len(items[show])} '
          f'(same-gender {g["same"]}, cross-gender {g["cross"]})')
    for k, v in skip.most_common():
        print(f'   skipped {v:3d}: {k}')

print('\nName replacement check (X question -> Y question):')
for show in SHOWS:
    full = [it for it in items[show] if len(it['q_x']) - len(it['q_y']) != len(it['x']) - len(it['y'])]
    print(f'  {show}: {len(full)} of {len(items[show])} questions had a title/full name replaced')
    for it in (full[:4] + items[show][:2])[:6]:
        print(f"    {it['q_x']}\n      -> {it['q_y']}")
ex = items['friends'][0]
print('\nFull prompt example:')
print(build_prompt(ex['q_y'], ex['opts']))
print(f"  X correct = {LETTERS[ex['x_idx']]}, Y correct = {LETTERS[ex['y_idx']]}\n")
if '--dry' in sys.argv:
    sys.exit('Dry run only (no model loaded).')


def load_frames(vpath, n=N_FRAMES):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    arr = vr.get_batch(idx).numpy()
    return [Image.fromarray(arr[i]) for i in range(n)]


BLACK = [Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)) for _ in range(N_FRAMES)]


def extract(resp):
    return next((ch for ch in resp.strip().upper() if ch in LETTERS), 'X')


def run_model(folder, infer_fn):
    for show in SHOWS:
        out_path = RES / folder / f'predictions_valid_option_swap_{show}.json'
        results = json.load(open(out_path, encoding='utf-8')) if out_path.exists() else []
        want = {it['qid']: it['q_y'] for it in items[show]}
        results = [r for r in results if want.get(r['question_id']) == r.get('question_y')]
        done = {(r['question_id'], r['mode']) for r in results}
        for mode in MODES:
            todo = [it for it in items[show] if (it['qid'], mode) not in done]
            print(f'\n--- {folder} / {show} / {mode} --- done {len(items[show]) - len(todo)}, '
                  f'remaining {len(todo)}')
            for it in todo:
                try:
                    frames = BLACK if mode == 'black' else load_frames(os.path.join(BASE_DIR, it['video']))
                    rx = infer_fn(frames, build_prompt(it['q_x'], it['opts']))
                    ry = infer_fn(frames, build_prompt(it['q_y'], it['opts']))
                except Exception as e:
                    print(f"  ERROR Q{it['qid']}: {e}"); continue
                px, py = extract(rx), extract(ry)
                xl, yl = LETTERS[it['x_idx']], LETTERS[it['y_idx']]
                results.append(dict(
                    question_id=it['qid'], show=show, mode=mode, original_char=it['x'],
                    swap_char=it['y'], gender=it['gender'], x_label=xl, y_label=yl,
                    pred_x=px, pred_y=py, raw_x=rx, raw_y=ry, len_match=it['len_match'],
                    y_already_present=it['y_already_present'], y_source_qid=it['y_qid'],
                    question_y=it['q_y']))
                print(f"  Q{it['qid']} {mode:5s} {it['x']:>9}->{it['y']:<9} "
                      f"X-correct={xl} Y-correct={yl} | asked X: {px}  asked Y: {py}")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                torch.cuda.empty_cache()


if __name__ == '__main__':
    # ---------------- InternVL2 (identical to run_valid_option_swap.py) ----------------
    print('=' * 62); print('  InternVL2-8B'); print('=' * 62)
    from transformers import AutoTokenizer, AutoModel
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    tok = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)
    mdl = AutoModel.from_pretrained(INTERNVL_PATH, torch_dtype=torch.float16,
                                    device_map='cuda:0', trust_remote_code=True)
    mdl.eval(); print('loaded')

    def iv_infer(frames, prompt):
        pv = torch.stack([tf(f) for f in frames]).half().cuda()
        ftxt = '\n'.join(f'Frame{i+1}: <image>' for i in range(len(frames)))
        return mdl.chat(tok, pv, f'{ftxt}\n{prompt}', dict(max_new_tokens=10, do_sample=False))

    run_model('InternVL2-8B', iv_infer)
    del mdl, tok; torch.cuda.empty_cache()

    # ---------------- Qwen (identical to run_valid_option_swap.py) ----------------------
    print('=' * 62); print('  Qwen2.5-VL-7B'); print('=' * 62)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    qmdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16, device_map='cuda:0', trust_remote_code=True)
    qmdl.eval(); print('loaded')

    def qw_infer(frames, prompt):
        messages = [{'role': 'user', 'content':
                     [{'type': 'image', 'image': f} for f in frames] +
                     [{'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(messages)
        inputs = proc(text=[text], images=ii, videos=vi, padding=True,
                      return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = qmdl.generate(**inputs, max_new_tokens=10, do_sample=False)
        return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    run_model('Qwen25VL', qw_infer)
    print('\nALL DONE. Now run: python analyze_valid_option_swap_fc.py')
