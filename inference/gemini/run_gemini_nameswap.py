"""Gemini 2.5 Flash name swap on BBT: original and swapped question for each pair, skipping
pairs already in predictions_name_swap_full.json. Saves after every call (resumable).
Reads the API key from GEMINI_API_KEY.
"""
import json
import os
import time
import ast
import random
from pathlib import Path

API_KEY = os.environ.get('GEMINI_API_KEY', '')
MODEL = 'gemini-2.5-flash'

if not API_KEY:
    print('Set the GEMINI_API_KEY environment variable.')
    print('Get one from https://aistudio.google.com/app/apikey')
    raise SystemExit(1)

# Install: pip install google-genai
from google import genai
client = genai.Client(api_key=API_KEY)

ANNOT_BBT = Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json')
VIDEOS_ROOT = Path(r'D:\InifiniBenchVideos\train_videos')
GEMINI_DIR = Path(r'D:\eval_workspace\results\global_appearance\Gemini')
GEMINI_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = GEMINI_DIR / 'predictions_name_swap_full.json'
TARGET_N = 124  # match open-source sample size

BBT_CAST_MALE = ['Sheldon', 'Leonard', 'Howard', 'Raj']
BBT_CAST_FEMALE = ['Penny', 'Bernadette', 'Amy']
BBT_CAST = BBT_CAST_MALE + BBT_CAST_FEMALE


def find_character(question):
    q_lower = question.lower()
    for name in BBT_CAST:
        if name.lower() in q_lower:
            return name
    return None


def resolve_video(video_rel):
    for c in [VIDEOS_ROOT / video_rel,
              VIDEOS_ROOT / video_rel.replace('TV_shows/videos/', '')]:
        if c.exists():
            return c
    return None


def upload_video(video_path):
    file_size_mb = video_path.stat().st_size / 1024 / 1024
    print(f'  Uploading {video_path.name} ({file_size_mb:.1f} MB)...')
    try:
        file = client.files.upload(file=str(video_path))
    except Exception as e:
        print(f'    Upload exception: {type(e).__name__}: {e}')
        raise
    print(f'    Initial state: {file.state.name}')
    waited = 0
    while file.state.name == 'PROCESSING':
        time.sleep(3)
        waited += 3
        file = client.files.get(name=file.name)
        if waited % 30 == 0:
            print(f'    Still processing after {waited}s...')
        if waited > 180:
            print(f'    Processing timed out after {waited}s')
            raise RuntimeError(f'Processing timeout for {video_path}')
    if file.state.name == 'FAILED':
        print(f'    Final state: FAILED')
        print(f'    Error info: {getattr(file, "error", "no error attribute")}')
        raise RuntimeError(f'Processing failed for {video_path}')
    print(f'    Ready (state={file.state.name})')
    return file


def call_gemini(video_file, prompt, max_tokens=20):
    response = client.models.generate_content(
        model=MODEL,
        contents=[video_file, prompt],
        config={'max_output_tokens': max_tokens, 'temperature': 0.0},
    )
    return response.text.strip() if response.text else ''


def extract_letter(response_text):
    response = response_text.strip().upper()
    for ch in response:
        if ch in 'ABCDE':
            return ch
    return '?'


def format_options(options):
    lines = []
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        text = ', '.join(str(x) for x in opt) if isinstance(opt, list) else str(opt)
        lines.append(f'{letter}. {text}')
    return '\n'.join(lines)


# Load annotations
print('Loading annotations...')
with open(ANNOT_BBT) as f:
    annots = json.load(f)
print(f'  {len(annots)} BBT validation questions')

# Load existing
existing = []
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE) as f:
        existing = json.load(f)
done_qids = {str(r['question_id']) for r in existing}
print(f'  Already done: {len(existing)} pairs')
print(f'  Target:       {TARGET_N} pairs')
print(f'  Remaining:    ~{TARGET_N - len(existing)} pairs to run')

if len(existing) >= TARGET_N:
    print('\nAlready at target sample size. Nothing to do.')
    raise SystemExit(0)

# Find remaining candidates
candidates = []
for a in annots:
    qid = str(a['question_id'])
    if qid in done_qids:
        continue
    orig_char = find_character(a.get('question', ''))
    if not orig_char:
        continue
    video_path = resolve_video(a.get('video_path_mp4', ''))
    if not video_path:
        continue
    candidates.append((qid, orig_char, video_path, a))

print(f'  Eligible candidates remaining: {len(candidates)}\n')

if not candidates:
    print('No remaining candidates. Done.')
    raise SystemExit(0)

# Confirm before spending money
n_to_run = min(TARGET_N - len(existing), len(candidates))
est_cost_low = n_to_run * 2 * 0.10
est_cost_high = n_to_run * 2 * 0.20
print(f'About to run {n_to_run} pairs ({n_to_run * 2} API calls)')
print(f'Estimated cost: ${est_cost_low:.0f} - ${est_cost_high:.0f}')
print()
response = input('Continue? (y/n): ')
if response.lower() != 'y':
    raise SystemExit('Aborted.')

# Deterministic swap selection (same seed as your open-source script if known)
rng = random.Random(42)

for qid, orig_char, video_path, annot in candidates:
    if len(existing) >= TARGET_N:
        print(f'\nReached target n={TARGET_N}. Stopping.')
        break

    other_chars = [c for c in BBT_CAST if c != orig_char]
    swap_char = rng.choice(other_chars)

    try:
        options = ast.literal_eval(annot['options'])
    except (SyntaxError, ValueError):
        print(f'  QID {qid}: bad options, skipping')
        continue

    answer_idx = int(annot.get('answer_idx', 0))
    correct_letter = chr(ord('A') + answer_idx)

    try:
        video_file = upload_video(video_path)
    except Exception as e:
        print(f'  QID {qid}: upload failed: {e}')
        continue

    question = annot['question']
    opt_text = format_options(options)

    orig_prompt = f'{question}\n\n{opt_text}\n\nAnswer with only the letter (A/B/C/D/E).'
    swap_question = question.replace(orig_char, swap_char)
    swap_prompt = f'{swap_question}\n\n{opt_text}\n\nAnswer with only the letter (A/B/C/D/E).'

    try:
        orig_resp = call_gemini(video_file, orig_prompt)
        swap_resp = call_gemini(video_file, swap_prompt)
    except Exception as e:
        print(f'  QID {qid}: API error: {e}')
        time.sleep(5)  # back off briefly
        continue

    orig_pred = extract_letter(orig_resp)
    swap_pred = extract_letter(swap_resp)

    record = {
        'question_id': qid,
        'original_char': orig_char,
        'swap_char': swap_char,
        'correct_label': correct_letter,
        'orig_pred': orig_pred,
        'swap_pred': swap_pred,
        'orig_correct': orig_pred == correct_letter,
        'answer_changed': orig_pred != swap_pred,
    }
    existing.append(record)

    # Save after every call so we don't lose progress
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(existing, f, indent=2)

    flag = 'changed' if record['answer_changed'] else 'same'
    print(f'  [{len(existing)}/{TARGET_N}] QID {qid} {orig_char[:4]}->{swap_char[:4]}: '
          f'{orig_pred}->{swap_pred} ({flag})')

    time.sleep(1)  # polite pacing

print(f'\nFinished. Total pairs: {len(existing)}')
print(f'Saved to: {OUTPUT_FILE}')
