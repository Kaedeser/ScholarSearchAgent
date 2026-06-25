# 爬虫策略模型调参分析

## Baseline 情况

已完成的 baseline 训练配置：

```text
model: Qwen/Qwen2.5-3B-Instruct
framework: LLaMA-Factory 0.9.5
method: LoRA SFT
lora_rank: 8
lora_alpha: 16
cutoff_len: 1024
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
GPUs: 7 x V100 16GB
effective_batch_size: 112
```

baseline 指标：

```text
train_loss: 0.3578266545578285
eval_loss final: 0.17109628021717072
eval_loss checkpoint-50: 0.17330560088157654
eval_loss checkpoint-100: 0.17636631429195404
eval_loss checkpoint-150: 0.17080816626548767
```

baseline 的验证 loss 在较早阶段就趋于平稳，说明继续简单增加 epoch 的收益有限。

## 数据和长度分析

数据处理结果：

```text
raw lines: 12989
kept: 9494
train: 8994
eval: 500
skipped_missing_sections: 3011
skipped_too_many_expands: 484
```

token 长度统计：

```text
train mean: 509
train p95: 787
train p99: 988
train max: 1980
train >1024: 72 samples, 0.80%

eval mean: 511
eval p95: 800
eval p99: 1100
eval max: 1513
eval >1024: 8 samples, 1.60%
```

结论：

```text
cutoff_len=1024 不是主要瓶颈。
直接提高到 1536 只能覆盖很少样本，但会增加显存压力。
```

标签分布：

```text
0 expands: 4317
1 expand: 808
2 expands: 1280
3 expands: 1189
4 expands: 942
5 expands: 600
6 expands: 358
```

数据中 StopExpand 样本较多，因此不能只看 token loss，还需要动作级指标。

## 是否可以通过调参提升

可以，但主要方向不是简单增加训练轮数。

baseline 的限制主要有三点：

```text
1. effective_batch_size=112 较大，总优化步数只有 162，更新粒度偏粗。
2. lora_rank=8 比较保守，对 section 名称和结构化动作生成的表达能力有限。
3. eval_loss 不能完全代表爬虫策略质量，需要 exact_match、parse_success、stop_accuracy、section_f1 等动作级指标。
```

## 已执行的优化方案

本轮采用低风险的 3B LoRA 优化：

```text
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 3
gradient_accumulation_steps: 8
cutoff_len: 1024
save_steps: 50
eval_steps: 50
```

预期效果：

```text
优化步数从 162 增加到 483。
LoRA 参数容量增加。
学习率从 1e-4 降到 5e-5，降低 3 epoch 训练下的过拟合风险。
```

实际结果：

```text
eval_loss: 0.171096 -> 0.159352
section_f1: 0.2920 -> 0.3007
parse_success_rate: 0.924 -> 0.986
```

说明本轮参数优化有效。

## 模型选择结论

训练过程中 step350 的 eval_loss 最低：

```text
step350 eval_loss: 0.156593
```

但动作级评估中，最终模型的 `section_f1` 更高：

```text
tuned final section_f1: 0.3007
tuned step350 section_f1: 0.2959
```

因此当前推荐使用最终模型：

```text
crawler_strategy_project/outputs/qwen2p5-3b-crawler-lora-r16-e3
```

## 后续调参建议

下一步不建议直接继续增加 epoch。更有价值的方向是：

```text
1. 调整 StopExpand / Expand 数据比例，减少模型过度偏向停止或展开。
2. 增加真实检索链路上的端到端评估，而不只看 SFT eval。
3. 对输出动作加规则校验，例如限制 section 必须来自输入列表。
4. 如果 3B 仍不够，再考虑 Qwen2.5-7B QLoRA。
```
