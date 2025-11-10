- 训练概况：已以 15s 时长与标签静音掩码进行 6-epoch LoRA 训练（rank=32, alpha=64, lr=1e-4, grad_accum=8），输出在 `finetune/output_full`。此前 1-epoch sanity 训练正常收敛（无 NaN），验证也正常。
- 评测改造：
  - `evaluate.py` 新增多模型对比函数 `compare_multi_models`，支持任意 1～3 个模型（`base`、`best`、`final` 等自定义 key）并将各自输出保存到 `evaluation/<key>_model/` 子目录，统一写入 JSONL（每行含 `<key>_model_file`）。
  - `batch_evaluate.py` 接受 `--best_model_path`、`--final_model_path`（可选）并调用多模型对比；仍支持只对比 base + lora 或单模型评测。
- 已执行批量评测（9 个 prompt）：
  - 命令已运行，输出目录：
    - 音频与结果：`/root/autodl-tmp/musicgen/finetune/output_full/evaluation/all_models/`
    - 各模型音频子目录：`base_model/`、`best_model/`、`final_model/`
    - 汇总 JSONL：`comparison_results.jsonl`

如果需要，我可以再跑一次使用完整 `evaluation_prompts_9.json` 的 base/best/final 全量评测报告整合出一个简短可视化表（含文件链接）。


分析训练过程，并确认批量评测结果。



检查训练日志和训练输出，分析训练过程：



检查训练是否完成，并验证模型文件：



基于现有信息，整理训练与评测分析报告：



## 1. 训练过程分析

### 训练配置（6-epoch 全量训练）
- 模型路径：`/root/autodl-tmp/musicgen/local/musicgen-small`
- 输出目录：`/root/autodl-tmp/musicgen/finetune/output_full`
- 训练参数：
  - LoRA rank: 32（此前为 12）
  - LoRA alpha: 64（此前为 24）
  - LoRA dropout: 0.05（此前为 0.1）
  - 学习率: 1e-4（此前为 3e-4）
  - Batch size: 1
  - Gradient accumulation: 8
  - Epochs: 6
  - Warmup steps: 200
  - 音频时长: 15s（此前为 30s，已修复 padding 问题）

### 训练结果
- 模型保存时间：
  - best_model: 2025-11-10 21:55
  - final_model: 2025-11-10 21:57
- 模型文件：
  - LoRA adapter: ~79MB（`adapter_model.bin`）
  - 配置：rank=32, alpha=64, dropout=0.05
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`

### 关键改进
1. 标签掩码：基于 `padding_mask` 将静音段标签设为 -100，避免学习静音
2. 时长统一：训练与推理均使用 15s，避免过度 padding
3. 超参调整：降低学习率、提高 rank/alpha，提升稳定性