# 重新进行LoRA微调与评测策略
_Exported on 2025/11/11 at GMT+8 0:32:34 from Cursor (2.0.69)_

---

**User**

之前已经经历过4次LoRA微调（有@output @output_fix @output_fix2 @output_full ），效果一般。



经过判断是之前的训练数据和验证数据有一些问题，现在修复后，需要对模型重新进行LoRA微调。元数据与数据集在@metadata.json @train @valid 



同时，你需要搜索与参考网络上的musicgen-small的LoRA教程与案例，以便你进行参数的配置等。训练框架与评测框架都已经搭建好@lora 。可参考的网址有：https://huggingface.co/danhtran2mind/MusicGen-Small-MusicCaps-finetuning https://huggingface.co/blog/theeseus-ai/musicgen-lora-large https://replicate.com/blog/fine-tune-musicgen https://huggingface.co/models?other=base_model:finetune:facebook/musicgen-small 



训练的音频与最后评测的音频时长均为15s，评测的时候需要同时评测base_model、best_model、final_model。



需要先在1epoch训练后针对@evaluation_prompts_8.json 评测一次，用户对音频文件进行主观判断后以及训练日志的分析后再调整训练策略，如果效果不错再考虑增加epoch。

---

**Cursor**

I’m ready to set up a fresh 1‑epoch LoRA run on `facebook/musicgen-small`, then evaluate base/best/final at 15s. Before I draft the execution plan, I need two confirmations.

1. Where is `evaluation_prompts_8.json` located (full path under `/root/autodl-tmp/musicgen`)?  
2. Training hardware:  
   - a) Single CUDA GPU (default)  
   - b) Multi-GPU (DDP)  
   - c) CPU-only

References:
- Hugging Face MusicGen-small finetune card (useful LR/epochs/batch baselines): `https://huggingface.co/danhtran2mind/MusicGen-Small-MusicCaps-finetuning`
- LoRA workflow and API serving best practices (adaptable from Large to Small): `https://huggingface.co/blog/theeseus-ai/musicgen-lora-large`
- Practical fine-tuning guidance (data labeling, chunking, params, and QA): `https://replicate.com/blog/fine-tune-musicgen`
- MusicGen-small finetunes browser for additional baselines: `https://huggingface.co/models?other=base_model:finetune:facebook/musicgen-small`

---

**User**

在这@evaluation_prompts_8.json。单块4090，之前的训练都挺快的。

请开始制定plan 

---

**Cursor**

Let me quickly scan the LoRA scripts and docs in `finetune/lora` to align the plan with the exact CLI and config knobs, then I’ll draft the step-by-step plan with commands.

<!-- aaaef58a-0d56-48c1-98a2-cdeb910b4f42 59eebe72-e412-4f03-967d-d3d262ab74bb -->
# MusicGen-small LoRA retrain + 15s evaluation (1-epoch gate)

## Goals

- Re-train LoRA on fixed data for 1 epoch at 15s audio.
- Evaluate base_model, best_model, final_model on 8 prompts (15s) and review.
- Adjust hyperparameters or extend epochs only after this gate.

## References

- MusicGen-small LoRA baseline hyperparams and results: `https://huggingface.co/danhtran2mind/MusicGen-Small-MusicCaps-finetuning`
- Practical LoRA workflow (applies to Small as well): `https://huggingface.co/blog/theeseus-ai/musicgen-lora-large`
- End-to-end fine-tuning guidance and QA: `https://replicate.com/blog/fine-tune-musicgen`
- Other MusicGen-small finetunes for parameter inspiration: `https://huggingface.co/models?other=base_model:finetune:facebook/musicgen-small`

## Environment

- GPU: Single RTX 4090 (CUDA)
- Conda: run `conda activate music` before commands
- Project root: `/root/autodl-tmp/musicgen`

## Data

- Metadata: `finetune/data/metadata.json`
- Train: `finetune/data/train/`
- Valid: `finetune/data/valid/`
- Eval prompts (8): `finetune/data/eval/evaluation_prompts_8.json`
- Target duration for train/eval: 15s

## Steps

### 1) Setup and sanity checks (no changes)

- Activate env and install local deps:
- `conda activate music`
- `cd /root/autodl-tmp/musicgen/finetune/lora && bash install.sh`
- Quick smoke tests:
- `python test_dataset.py`
- `python test_model.py`

### 2) One-epoch LoRA fine-tune (15s)

- New output dir (example): `/root/autodl-tmp/musicgen/finetune/output_epoch1`
- Command (single GPU 4090, fp16, 15s clips):
- `cd /root/autodl-tmp/musicgen/finetune/lora`
- `python train_lora.py \
--model_path /root/autodl-tmp/musicgen/local/musicgen-small \
--data_dir ../data \
--metadata_file metadata.json \
--output_dir ../output_epoch1 \
--lora_rank 12 \
--lora_alpha 24 \
--learning_rate 1e-4 \
--batch_size 2 \
--gradient_accumulation_steps 4 \
--num_epochs 1 \
--warmup_steps 50 \
--save_steps 200 \
--logging_steps 10 \
--max_duration 15 \
--target_duration 15 \
--fp16`
- Notes:
- Start with lr=1e-4 (safer than 3e-4 for LoRA on Small). If underfitting, consider 2e-4; if unstable, 5e-5.
- Keep rank/alpha at 12/24 initially; adjust after gate if style not captured.

### 3) Post-epoch evaluation at 15s

- Compare base, best, final on 8 prompts:
- `python evaluate.py \
--base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
--lora_model_path ../output_epoch1 \
--best_model_path ../output_epoch1/best_model \
--final_model_path ../output_epoch1/final_model \
--test_prompts_file ../data/eval/evaluation_prompts_8.json \
--output_dir ../output_epoch1/evaluation \
--duration 15 \
--compare_multi`
- Outputs:
- Audio for each model under `../output_epoch1/evaluation/{base_model,best_model,final_model}/`
- `comparison_results.jsonl` with paths and prompts

### 4) Review gate (subjective + logs)

- Subjective listening on generated WAVs for base/best/final.
- Inspect training logs for:
- Valid loss vs. train loss (over/underfitting signals)
- Stability (loss spikes, NaNs)
- Steps per second and GPU utilization

### 5) Adjust strategy or extend epochs

- If quality improved but still generic: raise `lora_rank` to 16 and `lora_alpha` to 32; keep lr.
- If noisy/unstable: lower lr to 5e-5; keep rank/alpha.
- If underfitting: increase epochs to 3–5, keep 15s duration.
- If rhythm/timbre off: consider `gradient_accumulation_steps=8` with `batch_size=1` to stabilize.

### 6) (Optional) Next training round

- Use new output dir (e.g., `output_fix3`), keep same eval flow, then decide on longer runs.

## Deliverables

- 1-epoch LoRA artifacts in `finetune/output_epoch1/{best_model,final_model,checkpoints,logs}`
- 15s evaluation WAVs for base/best/final and `comparison_results.jsonl`
- Brief notes with subjective findings + log snapshots

### To-dos

- Activate conda, install deps under finetune/lora
- Sanity check metadata and dataset loading at 15s
- Run 1-epoch LoRA training to output_epoch1
- Evaluate 15s prompts across base/best/final into evaluation dir
- Listen to outputs and inspect logs for stability/fit
- Propose next hyperparams or extend epochs based on findings
---
