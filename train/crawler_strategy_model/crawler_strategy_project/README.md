# Crawler Strategy Model Training

This directory wraps LLaMA-Factory 0.9.5 for the PaSa crawler strategy SFT task.

Chosen route:

- framework: LLaMA-Factory
- base model: `Qwen/Qwen2.5-3B-Instruct`
- method: LoRA SFT
- data format: original PaSa OpenAI `messages`
- output format: `[Expand]section title[Expand]section title[StopExpand]`
- hardware profile: P100 16GB, fp16, bf16 disabled, flash attention disabled

Local layout:

- `../LLaMA-Factory/`: unpacked `llamafactory==0.9.5` source package.
- `packages/`: local wheel used by the remote virtual environment.
- `data/raw/sft_crawler_train.jsonl`: original PaSa crawler SFT data.
- `data/dataset_info.json`: LLaMA-Factory dataset registry.
- `configs/crawler_qwen2p5_3b_lora.yaml`: default training config.
- `configs/crawler_qwen2p5_3b_lora_r16_e3.yaml`: tuned Round 1 config from the parameter analysis.
- `scripts/prepare_crawler_data.py`: validates and splits train/eval data.
- `scripts/train_cu05.sh`: creates a virtual environment and starts training.
- `scripts/start_training_background.sh`: starts training with `nohup`.
- `scripts/deploy_to_cu05.py`: deploys through the mu01 jump host without storing credentials.
- `scripts/run_action_eval.sh`: generates eval predictions and computes crawler action metrics.

Prepare data locally:

```bash
python scripts/prepare_crawler_data.py --source data/raw/sft_crawler_train.jsonl --out-dir data --eval-size 500
```

Train on cu05:

```bash
bash scripts/train_cu05.sh
```

Start in background on cu05:

```bash
bash scripts/start_training_background.sh
tail -f logs/start_*.log
```

Start the tuned Round 1 run:

```bash
CONFIG_PATH=configs/crawler_qwen2p5_3b_lora_r16_e3.yaml bash scripts/start_training_background.sh
```

Evaluate the tuned adapter with action-level metrics:

```bash
ADAPTER_DIR=outputs/qwen2p5-3b-crawler-lora-r16-e3 bash scripts/run_action_eval.sh
```

The large `cs_paper_2nd.zip` auxiliary corpus is intentionally not copied into this package because the first SFT milestone only uses `sft_crawler/train.jsonl`. Add it later for retrieval-level crawler evaluation.
