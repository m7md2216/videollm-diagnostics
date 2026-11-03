"""Gemini 2.5 Flash open-ended answers on the same BBT questions as the open-source models.
Writes predictions_openended.json with the ground truth attached for judging. Stops at a
fixed API budget; resumable. Reads the API key from GEMINI_API_KEY.
"""
import os, json, ast, time, sys
from pathlib import Path
from google import genai
from google.genai import types

# ---- API setup ----
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise SystemExit('Set GEMINI_API_KEY first.')

# Hard cost cap
MAX_TOTAL_USD = 20.00

# Gemini 2.5 Flash pricing (per 1M tokens)
PRICE_INPUT_PER_1M  = 0.30
PRICE_OUTPUT_PER_1M = 2.50

client   = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-2.5-flash'

# ---- Paths ----
VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'

# Source files for matched question IDs (we use Qwen's 52 as the canonical set)
QWEN_OE_FILE     = r'D:\eval_workspace\results\global_appearance\Qwen25VL\predictions_openended.json'
INTERNVL_OE_FILE = r'D:\eval_workspace\results\global_appearance\InternVL2-8B\predictions_openended.json'

# Gemini output directory
OUT_DIR = Path(r'D:\eval_workspace\results\global_appearance\Gemini')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE  = OUT_DIR / 'predictions_openended.json'
COST_FILE = OUT_DIR / 'cost_log_openended.json'

# ---- Load matched question IDs from existing open-source open-ended runs ----
print('Loading matched question IDs from existing open-source runs...')
target_qids = set()
for f in [QWEN_OE_FILE, INTERNVL_OE_FILE]:
    if not os.path.exists(f):
        print(f'  WARNING: {f} not found, skipping')
        continue
    with open(f) as fh:
        data = json.load(fh)
    for r in data:
        target_qids.add(str(r['question_id']))
print(f'Matched {len(target_qids)} unique question IDs from open-source open-ended runs\n')

# ---- Load annotation and filter to matched IDs ----
with open(VAL_ANN) as f:
    val_data = [d for d in json.load(f) if d.get('is_visual', False)]
val_by_id = {str(d['question_id']): d for d in val_data}

to_run = [val_by_id[qid] for qid in sorted(target_qids) if qid in val_by_id]
print(f'Will run {len(to_run)} Gemini open-ended questions\n')

# ---- Cost tracking (resume-safe) ----
cost_log = {'total_usd': 0.0, 'queries': []}
if COST_FILE.exists():
    with open(COST_FILE) as f:
        cost_log = json.load(f)
    print(f'Resuming cumulative cost: ${cost_log["total_usd"]:.4f}\n')

def save_cost():
    with open(COST_FILE, 'w') as f:
        json.dump(cost_log, f, indent=2)

def record_cost(qid, in_tok, out_tok):
    c_in  = in_tok  * PRICE_INPUT_PER_1M  / 1_000_000
    c_out = out_tok * PRICE_OUTPUT_PER_1M / 1_000_000
    total = c_in + c_out
    cost_log['total_usd'] += total
    cost_log['queries'].append({
        'qid': qid, 'input_tokens': in_tok,
        'output_tokens': out_tok, 'cost_usd': total,
    })
    save_cost()
    return total

def check_budget():
    if cost_log['total_usd'] >= MAX_TOTAL_USD:
        print(f'\n*** BUDGET CAP REACHED: ${cost_log["total_usd"]:.2f} ***')
        raise SystemExit('Budget cap reached')

# ---- Helpers ----
def parse_options(opts_field):
    try:
        return ast.literal_eval(opts_field)
    except:
        return eval(opts_field)

def parse_gap(tg):
    try:
        ts = ast.literal_eval(tg)
        starts = []
        for group in ts:
            if isinstance(group, list) and isinstance(group[0], list):
                for w in group:
                    h,m,s = w[0].split(':')
                    starts.append(int(h)*3600+int(m)*60+float(s))
            elif isinstance(group, list) and isinstance(group[0], str):
                h,m,s = group[0].split(':')
                starts.append(int(h)*3600+int(m)*60+float(s))
        return (max(starts)-min(starts))/60 if len(starts)>=2 else 0
    except:
        return 0

def get_ground_truth_text(item):
    """Extract the actual text of the correct option."""
    label_chars = 'ABCDE'
    opts = parse_options(item['options'])
    answer = item['answer']
    if answer in label_chars and label_chars.index(answer) < len(opts):
        opt = opts[label_chars.index(answer)]
        if isinstance(opt, list):
            return ', '.join(opt)
        return str(opt)
    return str(answer)

def build_oe_prompt(item):
    """Open-ended prompt - matches what InternVL2/Qwen received."""
    return (
        f"Watch the full video carefully and track each character's appearance.\n\n"
        f"Question: {item['question']}\n\n"
        f"Answer in detail. Describe each outfit in the order it appears. "
        f"Be specific about colors, clothing items, and layering."
    )

def upload_video(vpath):
    vf = client.files.upload(file=vpath)
    while vf.state.name == 'PROCESSING':
        time.sleep(5)
        vf = client.files.get(name=vf.name)
    if vf.state.name == 'FAILED':
        raise Exception('Video processing failed')
    return vf

def extract_text(resp):
    try:
        if resp.text:
            return resp.text.strip()
    except:
        pass
    try:
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if getattr(part, 'text', None):
                    return part.text.strip()
    except:
        pass
    return ''

def ask_gemini(video_file, prompt):
    config = types.GenerateContentConfig(
        max_output_tokens=2048,  # open-ended needs more tokens than MC
        temperature=0.0,
    )
    resp = client.models.generate_content(
        model=MODEL_ID, contents=[video_file, prompt], config=config,
    )
    text = extract_text(resp)
    in_tok = 0
    out_tok = 0
    try:
        usage = resp.usage_metadata
        in_tok  = getattr(usage, 'prompt_token_count', 0) or 0
        out_tok = getattr(usage, 'candidates_token_count', 0) or 0
    except:
        pass
    return text, in_tok, out_tok

def is_quota_error(e):
    s = str(e).lower()
    return ('429' in s or 'quota' in s or 'resource exhausted' in s
            or 'rate limit' in s or 'exceeded' in s)

# ---- Load existing results (resume) ----
done = set()
results = []
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        results = json.load(f)
    done = {str(r['question_id']) for r in results}
    print(f'Already done: {len(done)} questions')

remaining = [item for item in to_run if str(item['question_id']) not in done]
print(f'Remaining: {len(remaining)} questions\n')

# ---- Main loop ----
print('=' * 60)
print('  GEMINI 2.5 FLASH - OPEN-ENDED BBT EVALUATION')
print('=' * 60)
start = time.time()

for item in remaining:
    check_budget()
    qid = str(item['question_id'])
    vpath = os.path.join(BASE_DIR, item['video_path_mp4'])
    prompt = build_oe_prompt(item)
    gt_text = get_ground_truth_text(item)
    gap = parse_gap(item['temporal_grounding'])
    gb = ('small' if gap<5 else 'medium' if gap<10
          else 'large' if gap<15 else 'xlarge')

    model_answer = ''
    attempt = 0
    max_retries = 5

    while attempt < max_retries:
        try:
            vf = upload_video(vpath)
            model_answer, in_tok, out_tok = ask_gemini(vf, prompt)
            record_cost(qid, in_tok, out_tok)
            try:
                client.files.delete(name=vf.name)
            except:
                pass
            break
        except Exception as e:
            if is_quota_error(e):
                attempt += 1
                print(f'  Q{qid}: quota hit, waiting 60s and retrying ({attempt}/{max_retries})')
                time.sleep(60)
                continue
            print(f'  Q{qid}: ERROR {e}')
            model_answer = f'ERROR: {e}'
            break

    results.append({
        'question_id':   qid,
        'question':      item['question'],
        'correct_label': item['answer'],
        'ground_truth':  gt_text,
        'model_answer':  model_answer,
        'judge_score':   None,
        'gap_minutes':   round(gap, 2),
        'gap_bucket':    gb,
    })
    preview = model_answer[:100].replace('\n', ' ')
    print(f'  Q{qid}: {preview}... | total=${cost_log["total_usd"]:.2f}')

    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    time.sleep(2)

# ---- Summary ----
elapsed = (time.time() - start) / 60
n_total = len(results)
n_error = sum(1 for r in results if r['model_answer'].startswith('ERROR'))
n_valid = n_total - n_error

print('\n' + '=' * 60)
print('  OPEN-ENDED EVALUATION COMPLETE')
print('=' * 60)
print(f'  Total rows:          {n_total}')
print(f'  Valid answers:       {n_valid}')
print(f'  Error answers:       {n_error}')
print(f'  Total cost:          ${cost_log["total_usd"]:.4f}')
print(f'  Elapsed:             {elapsed:.1f} min')
print(f'  Output:              {OUT_FILE}')
print()
print('  Next step: score predictions_openended.json with judge/judge_openended.py.')
