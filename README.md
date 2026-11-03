# Video-LLM Answers Follow Gender More Than Identity: code and outputs

Code, prompts and all model and judge outputs for the paper *"Video-LLM Answers Follow
Gender More Than Identity: Diagnosing Character Tracking in Long-Form Video"* (anonymous
submission).

**The test.** Video-LLMs answer InfiniBench's outfit-ordering questions about named TV
characters above chance. To check whether their answers follow the character a question
names, we keep the video (16 uniformly sampled frames) and the answer options fixed and
change only the named character. In the **valid-option swap**, the option set contains
both characters' correct answers, and a pair counts as correct only if the model answers
correctly for both characters. A black-frame control separates what the video adds from
what the names and option text alone allow.

## Repository layout

```
inference/          scripts that run the models (InternVL2-8B, Qwen2.5-VL-7B, LLaVA-NeXT-Video-7B)
inference/gemini/   Gemini 2.5 Flash runs (API key read from GEMINI_API_KEY)
analysis/           read-only analyses of the prediction files (no GPU)
judge/              open-ended scoring: rubric and LLM-judge runner
figures/            scripts that draw the paper's figures (figures/data holds the timeline data)
results/            per-question predictions for every model and condition, judge outputs,
                    name_swap_manifest.csv and MANIFEST.txt
```

## Where each result comes from

| Paper | What | Scripts | Result files (`results/global_appearance/<model>/`) |
|---|---|---|---|
| §4.1, Table 2 | Accuracy with video, black frames, show title only | prompts in Appendix B; `inference/rerun_llava_textonly.py`; `analysis/analyze_clustered_extra.py` (episode-level video − black) | `predictions.json`, `predictions_textonly*.json`, `../TextOnly/` |
| §4.2, Table 3, Tables 11, 16 | Name swap (answer-change rate), corrected wording | `inference/run_expanded_nameswap.py`, `inference/rerun_nameswap_fc_clean.py`; `analysis/analyze_swap_accuracy.py`, `analysis/analyze_nameswap_clean.py`, `analysis/check_swap_wording.py` | `predictions_name_swap_*.json` |
| §4.2 (Gemini) | 14 matched / 20 parseable swaps | `inference/gemini/*`; `analysis/gemini_conditioned_and_matched.py` | `../Gemini/` |
| §4.3, Figures 1–3, Tables 12–14 | Valid-option swap, video vs. black frames, same vs. cross gender, episode bootstrap | `analysis/count_valid_option_feasibility.py`, `inference/run_valid_option_swap.py` (BBT), `inference/run_valid_option_swap_fc.py` (Friends, Castle); `analysis/analyze_valid_option_swap*.py`, `analysis/check_friends_castle_gender.py`, `analysis/analyze_robustness.py` (Part A) | `predictions_valid_option_swap*.json` |
| Table 4, Table 15 | Robustness: InternVL2 documented input, frames from both characters' scenes, labels 1–5, reversed order | `inference/run_robustness.py`; `analysis/analyze_robustness.py` (Parts B–D) | `r2_*.json` |
| §4.4, Table 5, Tables 17–18, Figures 5–6 | Frame count, padded oracle, subtitles, pair-oracle gains | prompts in Appendix B; `inference/run_llava_64frames.py`, `inference/run_subtitles.py`, `analysis/compare_subtitle_results.py`, `analysis/analyze_clustered_extra.py` (difference in gains) | `predictions_frames_32.json`, `predictions_32frames.json`, `predictions_64frames.json`, `predictions_subtitle*.json`, `r2_pair_oracle_validopt.json` |
| §4.5, Figure 4, Table 10, judge rows of Table 9 | Open-ended answers and LLM judge | `inference/run_openended.py`, `inference/gemini/run_gemini_openended.py`; `judge/export_openended_for_judge.py`, `judge/judge_openended.py` with `judge/rubric.txt`; `analysis/check_long_answers.py` | `predictions_openended*.json`, `results/judge_outputs/` |
| §4.6, Tables 19–20, Figure 7 | Answer selection: letters, shuffled and 1–5 options, abstention | prompts in Appendix B; `analysis/check_baseline_idk.py` | shuffle / relabel / abstention prediction files |
| Figures | All figures | `figures/make_fig1.py` (needs four TVQA frames, not included), `make_fig2.py`, `make_figs_extra.py`, `make_fig_timeline.py` | `figures/data/timeline_data.json` |

### Superseded files

`results/` also keeps a few earlier runs that the paper replaced (Appendices B and C describe why).
The paper uses the files on the right:

| Earlier run | Used in the paper |
|---|---|
| `predictions_name_swap_{friends,castle}.json`, `*_expanded.json` (replacement name inserted incompletely) | `predictions_name_swap_{friends,castle}_clean.json` |
| `predictions_openended.json`, `predictions_openended_judged.json` (100–150-token answers, earlier judge) | `predictions_openended_v2.json`, scored in `rejudge_results_v2.json` |
| `predictions_subtitle.json` | `predictions_subtitle_v2.json` |
| `predictions_textonly.json` (LLaVA) | identical to its documented rerun `predictions_textonly_v2.json` |

## Setup

`requirements.txt` lists the full environment used. Inference ran on one NVIDIA RTX 3090
(24 GB) in float16 with greedy decoding, using PyTorch 2.5.1 (CUDA 12.1), Transformers
4.51.0 and decord 0.6.0.

The scripts were run on one machine and keep that machine's path constants at the top of
each file (for example `D:\eval_workspace\results\global_appearance`, `D:\models\...`).
Edit those constants to point at your copies of the InfiniBench annotations, the episode
videos and the model checkpoints. The analysis scripts only need the JSON files in
`results/`.

InternVL2 note: the original InternVL2 runs pass all 16 frames through a single image slot.
The runs in `inference/run_robustness.py` use InternVL2's documented video input (one
`<image>` slot per frame with `num_patches_list`). Appendix B of the paper lists which
results use which.

## Data

Annotations come from the InfiniBench release (`Vision-CAIR/InfiniBench`, revision
`1a46b0b303303515fa5872e3704c8b27a8ab7b0a`, BSD 3-Clause). The episode videos come from
TVQA and are provided for research under an agreement not to share the frames. This
repository therefore contains **no video, frames or images**: only question IDs, the
modified option sets, prompts and model and judge outputs. Obtain the videos through
InfiniBench and TVQA.

## License

Code: MIT (see `LICENSE`). Model outputs are released for research use; the underlying
annotations remain under InfiniBench's licence.
