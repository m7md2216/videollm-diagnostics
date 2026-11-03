"""Valid-option name swap on BBT.

For each BBT pair (X -> Y) from predictions_name_swap_full.json where Y has their
own global-appearance question in the same episode (valid_option_candidates.json):

  Build ONE modified option set, used in both conditions:
     original options[:5], with ONE distractor (not X's gold, not IDK) replaced by
     Y's correct outfit sequence. X's correct answer stays where it was.

  Condition X : original question (asks about X)  -> correct = X's option
  Condition Y : swapped question (asks about Y)    -> correct = Y's option

  Same frames, same options, same positions; only the NAME differs.
  A model that tracks identity moves from X's option to Y's option.
  A model that ignores the name gives the same answer in both.

MODES
  video : 16 uniform frames, as in the full-episode condition
  black : 16 black frames (language-prior control -- can the name alone pick
          Y's option, e.g. via name->outfit stereotypes?)

Output (one file per model, resumable):
  results/global_appearance/<model>/predictions_valid_option_swap.json
Nothing existing is overwritten.
"""
import os, json, ast, random, re, torch, numpy as np
from pathlib import Path
from PIL import Image
import decord
decord.bridge.set_bridge('torch')

ANN_PATH = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
INTERNVL_PATH = r'D:\models\internvl2_8b'
QWEN_PATH     = r'D:\models\qwen25_vl_7b'
RES = Path(r'D:\eval_workspace\results\global_appearance')
CANDS = RES / 'valid_option_candidates.json'

MODES = ['video', 'black']     # run video first; black is the language-prior control
SEED = 2026
N_FRAMES = 16
LETTERS = list('ABCDE')

ann = {str(d['question_id']): d for d in json.load(open(ANN_PATH, encoding='utf-8'))}
cand_rows = json.load(open(CANDS, encoding='utf-8'))['existing_pairs']['InternVL2-8B']


def get_options(item):
    try:
        return ast.literal_eval(item['options'])
    except Exception:
        return eval(item['options'])


def opt_text(opt):
    return ', '.join(opt) if isinstance(opt, list) else str(opt)


def is_idk(opt):
    return "know" in opt_text(opt).lower()


def swapped_question(q, x, y):
    return re.sub(rf'\b{re.escape(x)}\b', y.title(), q, flags=re.IGNORECASE)


def build_prompt(question, opts):
    options_text = ''.join(f'\n{LETTERS[i]}. {opt_text(o)}' for i, o in enumerate(opts))
    return (f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {question}\n"
            f"Options:{options_text}\n\n"
            f"Reply with ONLY a single letter: {', '.join(LETTERS[:len(opts)])}.")


# ---------------------------------------------------------------- build items
items = []
skipped = {}
for c in cand_rows:
    qid = c['qid']
    a = ann.get(qid)
    if a is None:
        skipped[qid] = 'no annotation'; continue
    opts = get_options(a)
    gx = int(a['answer_idx'])
    if gx > 4 or len(opts) < 5:
        skipped[qid] = f'gold index {gx} outside A-E'; continue
    opts = opts[:5]
    distractors = [i for i in range(5) if i != gx and not is_idk(opts[i])]
    if not distractors:
        skipped[qid] = 'no replaceable distractor'; continue
    rng = random.Random(f'{SEED}-{qid}')          # per-question, order-independent
    ry = rng.choice(distractors)
    new_opts = list(opts)
    new_opts[ry] = c['y_gold']
    items.append(dict(
        qid=qid, x=c['x'], y=c['y'], video=a['video_path_mp4'],
        q_x=a['question'], q_y=swapped_question(a['question'], c['x'], c['y']),
        opts=new_opts, x_idx=gx, y_idx=ry,
        len_match=(c['y_n_outfits'] == c['x_n_outfits'])))

print(f'Usable pairs: {len(items)}   skipped: {len(skipped)} {skipped}')
bad_swap = [it['qid'] for it in items if it['q_x'] == it['q_y']]
if bad_swap:
    print(f'WARNING: name not found in question for {bad_swap} -- these will be skipped')
    items = [it for it in items if it['q_x'] != it['q_y']]
ex = items[0]
print('\nExample (check this looks right):')
print('  X question:', ex['q_x'])
print('  Y question:', ex['q_y'])
print(build_prompt(ex['q_y'], ex['opts']))
print(f"  X correct = {LETTERS[ex['x_idx']]}, Y correct = {LETTERS[ex['y_idx']]}\n")


def load_frames(vpath, n=N_FRAMES):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    arr = vr.get_batch(idx).numpy()
    return [Image.fromarray(arr[i]) for i in range(n)]


BLACK = [Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)) for _ in range(N_FRAMES)]


def extract(resp):
    for ch in resp.strip().upper():
        if ch in LETTERS:
            return ch
    return 'X'


def run_model(folder, infer_fn):
    out_path = RES / folder / 'predictions_valid_option_swap.json'
    results = json.load(open(out_path, encoding='utf-8')) if out_path.exists() else []
    done = {(r['question_id'], r['mode']) for r in results}
    for mode in MODES:
        todo = [it for it in items if (it['qid'], mode) not in done]
        print(f'\n--- {folder} / {mode} --- done {len(items) - len(todo)}, remaining {len(todo)}')
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
                question_id=it['qid'], mode=mode, original_char=it['x'], swap_char=it['y'],
                x_label=xl, y_label=yl, pred_x=px, pred_y=py,
                raw_x=rx, raw_y=ry, len_match=it['len_match'],
                question_y=it['q_y']))
            print(f"  Q{it['qid']} {mode:5s} {it['x']:>10}->{it['y']:<10} "
                  f"X-correct={xl} Y-correct={yl} | asked X: {px}  asked Y: {py}")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # ---------------- InternVL2 (same loader/inference as run_option_shuffle.py) ----
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

    # ---------------- Qwen (same loader/inference as run_option_shuffle.py) ---------
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
