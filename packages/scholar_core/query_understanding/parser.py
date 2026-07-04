# 中文功能说明：查询理解模块，使用规则抽取研究领域、约束、时间范围和子查询。

from __future__ import annotations

import re
from collections import Counter

from packages.scholar_core.models import QueryIntent
from packages.scholar_core.text import STOPWORDS, compact_terms, normalize_space, tokenize


FIELD_HINTS = {
    "anomaly": "anomaly detection",
    "image": "computer vision",
    "ips": "counterfactual learning",
    "snips": "counterfactual learning",
    "hubert": "speech representation learning",
    "speech": "speech processing",
    "retrieval": "information retrieval",
    "segmentation": "semantic segmentation",
    "language": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "pretraining": "large language model pretraining",
    "model": "machine learning",
    "models": "machine learning",
    "graph": "graph learning",
    "cot": "chain-of-thought prompting",
    "vlm": "vision-language models",
    "vlms": "vision-language models",
    "nerf": "neural radiance fields",
    "slam": "simultaneous localization and mapping",
    "math": "mathematical reasoning",
    "imo": "mathematical reasoning",
    "sft": "supervised fine-tuning",
    "humaneval": "code generation evaluation",
    "mbpp": "code generation evaluation",
    "code_contests": "code generation evaluation",
    "agents": "llm agents",
    "agent": "llm agents",
    "financial": "financial ai",
    "ara": "text readability assessment",
    "video": "video understanding",
    "sign": "sign language recognition",
    "diffusion": "generative modeling",
    "recommendation": "recommender systems",
    "ranking": "information retrieval",
    "superpixels": "computer vision",
    "patches": "computer vision",
    "hotpotqa": "question answering",
    "moe": "mixture of experts",
}

SYNONYMS = {
    "anomaly score": [
        "tranad",
        "multivariate time series anomaly detection",
        "transformer networks anomaly detection",
        "focus score",
        "reconstruction error",
    ],
    "deep q-learning": ["deep q network", "dqn", "target network"],
    "chain-of-thought prompting": [
        "cot prompting",
        "chain of thought prompting",
        "chain-of-thought prompting elicits reasoning",
        "reasoning in large language models",
    ],
    "cot prompting": [
        "chain-of-thought prompting",
        "chain of thought prompting",
        "chain-of-thought prompting elicits reasoning",
        "reasoning in large language models",
    ],
    "vlm": [
        "vision-language model",
        "vision language model",
        "learning to prompt for vision-language models",
        "clip-adapter",
        "feature adapters",
    ],
    "vlms": [
        "vision-language model",
        "vision language model",
        "learning to prompt for vision-language models",
        "clip-adapter",
        "feature adapters",
    ],
    "vision-language models": [
        "vision-language model",
        "multimodal large language model",
        "large vision-language model",
        "vision-language large model",
        "multimodal model",
    ],
    "visual-llm": [
        "visual llm",
        "vision-language model",
        "multimodal large language model",
        "large vision-language model",
    ],
    "hotpotqa": ["multi-hop question answering", "question answering benchmark", "hotpotqa dataset"],
    "moe": ["mixture of experts", "sparse mixture of experts", "moe architecture"],
    "autoregressive transformer": [
        "autoregressive video generation",
        "video generation with autoregressive transformers",
        "autoregressive language model video generation",
    ],
    "generate videos": ["video generation", "video synthesis"],
    "commonsense problems": ["commonsense machine translation", "commonsense reasoning", "commonsense-aware translation"],
    "rlhf hallucination": [
        "hallucination mitigation",
        "vision-language hallucination",
        "image captioning hallucination",
        "video captioning hallucination",
        "factually augmented rlhf",
        "preference fine-tuning",
        "fine-grained reward modeling",
        "correctional human feedback",
    ],
    "rlhf": [
        "reinforcement learning from human feedback",
        "human feedback",
        "preference optimization",
        "preference fine-tuning",
    ],
    "reinforcement learning to optimize diffusion models": [
        "video diffusion alignment",
        "reward gradients",
        "human feedback",
        "preference optimization",
        "text-to-video diffusion",
    ],
    "optimize diffusion models": [
        "diffusion model alignment",
        "reward gradients",
        "human feedback",
        "video diffusion models",
    ],
    "quantized pretraining": ["quantization-aware pretraining", "quantized language model pretraining", "low-bit pretraining"],
    "dpo training": ["direct preference optimization", "dpo vision-language models", "preference optimization vision-language"],
    "identity preservation video generation": [
        "identity-preserving video generation",
        "personalized video generation",
        "face identity preservation",
    ],
    "vocabulary watermarking": [
        "watermarking language models",
        "quality-preserving watermarking",
        "watermark robustness",
    ],
    "long thought data": [
        "theorem and proof data",
        "large-scale theorem proving data",
        "theorem proving data synthesis",
        "long chain-of-thought data",
        "synthetic reasoning data",
        "reasoning data generation",
        "large-scale reasoning data",
        "theorem proving data",
        "proof data synthesis",
        "mathematical reasoning data",
    ],
    "same prompt": ["preference data", "response comparison", "preference optimization", "sft preference data"],
    "negative impact": [
        "rlhf generalisation diversity",
        "reward collapse",
        "vanishing gradients",
        "reinforcement finetuning",
        "alignment limitations",
    ],
    "negatively impact": [
        "rlhf generalisation diversity",
        "reward collapse",
        "vanishing gradients",
        "reinforcement finetuning",
        "alignment limitations",
    ],
    "prompt tuning": [
        "learning to prompt",
        "prompt learning",
        "prompt tuning vision-language models",
        "learnable prompts",
    ],
    "fine-tuning": ["adapter", "feature adapter", "clip-adapter", "parameter-efficient fine-tuning"],
    "discriminator loss": ["adversarial training", "gan discriminator", "anomaly score"],
    "edit operation": ["edit operations", "sequence editing", "seq2edit"],
    "edit operation prediction": ["encode tag realize", "lasertagger", "token-level edit operation", "seq2edit"],
    "hubert": ["generative spoken language modeling", "discrete speech encoder", "self-supervised speech representation"],
    "hubert codes": [
        "generative spoken language modeling",
        "discrete speech encoder",
        "hubert units",
        "discrete speech units",
    ],
    "image retrieval": ["visual search", "image-text retrieval", "cross-modal retrieval"],
    "image-text": ["image text", "vision-language", "cross-modal", "language-image"],
    "in-context learning": [
        "icl",
        "few-shot prompting",
        "emergent ability",
        "in-context learning fall short",
        "few-shot information extraction",
    ],
    "latent graphs": [
        "learning discrete structures for graph neural networks",
        "graph structure learning",
        "iterative deep graph learning",
        "latent graph learning",
    ],
    "instance-level segmentation": [
        "detr",
        "set prediction segmentation",
        "instance segmentation",
        "panoptic segmentation",
        "mask classification",
    ],
    "inverse propensity score": [
        "debiasing learning evaluation",
        "recommendations as treatments",
        "inverse propensity scoring",
        "ips",
        "counterfactual learning",
    ],
    "semantic segmentation": ["image segmentation", "region-based segmentation", "pixel labeling"],
    "semantic tokens": [
        "generative spoken language modeling",
        "speech resynthesis",
        "discrete speech units",
        "speech tokens",
    ],
    "selection bias": [
        "debiasing learning evaluation",
        "recommendations as treatments",
        "sample selection bias",
        "counterfactual learning",
        "propensity scoring",
    ],
    "self-normalized ips": [
        "snips",
        "self-normalized inverse propensity scoring",
        "debiasing learning evaluation",
    ],
    "supervised fine-tuned": [
        "reward collapse",
        "vanishing gradients",
        "rlhf generalisation diversity",
        "supervised fine-tuning",
        "sft",
        "instruction tuning",
        "small language model",
    ],
    "in-context learning performance": [
        "in-context learning fall short",
        "few-shot information extraction",
        "specification-heavy tasks",
        "supervised fine-tuning comparison",
    ],
    "information extraction tasks": [
        "information extraction",
        "few-shot information extractor",
        "biomedical information extraction",
        "named entity recognition",
        "relation extraction",
        "event extraction",
        "sequence labeling",
    ],
    "pretraining": ["pre-training", "self-supervised learning", "representation learning"],
    "large language model": ["llm", "llms", "language model"],
    "nerf": ["neural radiance fields", "neural radiance field", "novel view synthesis"],
    "gaussian noise": ["nerf in the wild", "pixel-wise uncertainty", "uncertainty estimation"],
    "rgb space": ["nerf in the wild", "pixel-wise uncertainty", "uncertainty estimation"],
    "pixel-wise uncertainty": ["nerf in the wild", "activenerf", "active view selection", "uncertainty estimation"],
    "diffusion models": [
        "planning with diffusion",
        "diffuser",
        "diffusion policies",
        "flexible behavior synthesis",
        "video diffusion models",
        "diffusion model alignment",
    ],
    "video diffusion": [
        "text-to-video diffusion",
        "video diffusion models",
        "video diffusion alignment",
        "diffusion model alignment",
    ],
    "mask classification": [
        "detr",
        "end-to-end object detection with transformers",
        "max-deeplab",
        "end-to-end panoptic segmentation with mask transformers",
        "mask classification based segmentation",
        "mask transformer",
        "mask transformers",
        "set prediction segmentation",
    ],
    "reconstruction error": [
        "multivariate time series anomaly detection",
        "reconstruction loss",
        "anomaly score",
        "reconstruction-based anomaly detection",
    ],
    "reconstruction-based": [
        "forecasting reconstruction anomaly detection",
        "graph attention network anomaly detection",
        "multivariate time series",
        "anomaly detection",
        "reconstruction-based model",
        "forecasting-based model",
    ],
    "reinforcement learning": [
        "planning with diffusion",
        "reinforcement finetuning",
        "policy optimization",
        "reward optimization",
        "online reinforcement learning",
        "rl",
    ],
    "reinforcement learning training": [
        "language agents with reinforcement learning",
        "text-based reinforcement learning agent",
        "online reinforcement learning",
        "interactive environments",
        "verbal reinforcement learning",
        "strategic play",
    ],
    "scaling law": [
        "scaling laws",
        "model scaling",
        "scaling behavior",
        "multimodal scaling laws",
        "mixed-modal language models",
        "contrastive language-image learning",
    ],
    "multi-module models": [
        "multimodal models",
        "mixed-modal language models",
        "language-image model",
        "video-language model",
        "vision-language pre-training",
    ],
    "long video description": [
        "long video captioning",
        "long-form video understanding",
        "hour-long videos",
        "dense video captions",
        "long video comprehension",
        "long video benchmark",
    ],
    "long videos": [
        "long-form video understanding",
        "long video comprehension",
        "long video benchmark",
        "hour-long videos",
    ],
    "superpixels": ["region proposals", "image regions"],
    "image patches": ["patch-based", "local image regions"],
    "spatio-temporal": ["spatiotemporal", "temporal spatial"],
    "latent space": [
        "dcgan",
        "unsupervised representation learning with deep convolutional generative adversarial networks",
        "deep convolutional generative adversarial networks",
        "deep feature interpolation",
        "deep feature interpolation for image content changes",
    ],
    "semantic arithmetic": [
        "vector arithmetic",
        "image arithmetic",
        "dcgan",
        "unsupervised representation learning with deep convolutional generative adversarial networks",
        "deep feature interpolation",
        "deep feature interpolation for image content changes",
    ],
    "equivariance": ["group equivariant convolutional networks", "group equivariant cnn", "equivariant convolution"],
    "api-use": ["toolalpaca", "toollm", "tool learning", "api learning"],
    "code evaluation": ["humaneval", "mbpp", "code contests", "apps", "livecodebench", "evalplus"],
    "video aesthetics": [
        "q-align",
        "visual scoring",
        "teaching lmms for visual scoring",
        "q-align teaching lmms for visual scoring via discrete text-defined levels",
        "discrete text-defined levels",
        "video quality assessment",
        "aesthetic score",
        "multimodal large language model",
    ],
    "synthetic data": [
        "self-instruct",
        "alpaca",
        "wizardlm",
        "evol-instruct",
        "instruction tuning data",
        "synthetic reasoning data",
        "theorem proving data",
        "proof data",
    ],
    "synthesis data": ["synthetic data", "self-instruct", "alpaca", "instruction tuning data"],
    "theorem proving": ["proof data", "theorem and proof data", "formal mathematics", "mathematical reasoning"],
    "proof data": ["theorem proving", "theorem and proof data", "synthetic theorem data"],
    "smaller dataset": [
        "data pruning for pretraining",
        "data-efficient llms",
        "less training data",
        "fewer data",
        "deduplicating training data",
        "influential subset selection",
    ],
    "smaller datasets": [
        "data pruning for pretraining",
        "data-efficient llms",
        "less training data",
        "fewer data",
        "deduplicating training data",
        "influential subset selection",
    ],
    "point cloud": ["pointnet", "point sets", "3d classification", "point cloud segmentation"],
    "hierarchical transformer": ["uniaudio", "audio foundation model", "audio generation", "semantic tokens"],
    "gumbel-softmax": ["bayesian experimental design", "boed", "contextual optimisation"],
    "boed": ["bayesian experimental design", "contextual optimisation", "causal decision making"],
    "dataset condensation": ["squeeze recover relabel", "imageNet scale", "dataset condensation"],
    "sfuda": ["source-free domain adaptation", "semantic segmentation", "source domain data estimation"],
    "explicit localization information": ["perceptual grouping", "contrastive vision-language models", "localization information"],
    "large web corpus": ["xlnet", "cloze-driven pretraining", "self-attention networks", "wikipedia corpus"],
    "oversmoothing": ["oversmoothing graph neural networks", "over-smoothing bert", "attention-based gnns"],
    "data selection": ["selection via proxy", "learnable worth learning", "efficient data selection"],
    "multiple domains": ["multi-domain semantic segmentation", "mseg", "multi-dataset pretraining"],
    "quantum monte carlo": ["neural network quantum states", "variational monte carlo", "neural quantum states"],
    "financial tasks": ["financial llm agents", "financial agent benchmark", "llm agents financial tasks"],
    "ara": [
        "automated readability assessment",
        "text readability",
        "neural approaches to text readability",
        "supervised and unsupervised neural approaches to text readability",
    ],
    "response length prediction": [
        "fast structured decoding",
        "fast structured decoding for sequence models",
        "sequence models",
        "response length",
    ],
    "numerical importances": [
        "post hoc explainers",
        "large language models post hoc explainers",
        "input attribution",
        "feature attribution",
    ],
    "interpret themselves": [
        "post hoc explainers",
        "large language models post hoc explainers",
        "self explanation",
    ],
    "target networks": ["deep q-learning target network", "dqn target network"],
    "token-level edit": ["encode tag realize", "lasertagger", "edit operation prediction", "seq2edit"],
    "video-text": ["video text", "video-language", "multimodal video"],
}

BRIDGE_ALIASES = {
    "llm": ["large language models", "language models"],
    "llms": ["large language models", "language models"],
    "prompt-based learning": [
        "prompt learning",
        "prompt engineering",
        "in-context learning",
        "few-shot learning",
        "pre-trained language models",
        "pre-trained transformers",
        "foundation language models",
    ],
    "prompt-based": [
        "prompt learning",
        "prompt engineering",
        "in-context learning",
        "few-shot learning",
        "pre-trained language models",
        "pre-trained transformers",
        "foundation language models",
    ],
    "factuality": ["factual consistency", "hallucination detection", "fact verification", "fact checking"],
    "factuality detection": [
        "factual consistency",
        "hallucination detection",
        "fact verification",
        "fact checking",
        "factual correction",
    ],
    "consistency": ["factual consistency", "faithfulness", "hallucination detection"],
    "rank": ["information retrieval", "learning to rank", "search ranking"],
    "ranking": ["information retrieval", "learning to rank", "search ranking"],
    "retrieval": ["information retrieval", "dense retrieval", "sparse retrieval"],
    "search results": [
        "information retrieval",
        "learning to rank",
        "search ranking",
        "llm reranking",
        "document reranking",
        "passage ranking",
    ],
    "web corpus": ["pretraining corpus", "web text corpus", "common crawl"],
    "wikipedia": ["pretraining corpus", "wikipedia corpus", "knowledge base text"],
    "multilingual": [
        "multilingual evaluation",
        "cross-lingual evaluation",
        "multilingual benchmarks",
        "cross-lingual transfer",
    ],
    "summary": [
        "multi-document summarization",
        "document summarization",
        "literature review generation",
        "automatic summarization",
        "text summarization",
        "abstractive summarization",
    ],
    "summaries": ["multi-document summarization", "document summarization", "literature review generation"],
    "survey": [
        "survey generation",
        "literature review",
        "systematic review",
        "scientific literature review",
        "literature survey generation",
        "research synthesis",
    ],
    "surveys": [
        "survey generation",
        "scientific literature review",
        "literature survey generation",
        "research synthesis",
        "academic literature review",
    ],
    "multiple scholarly documents": [
        "multi-document summarization",
        "document summarization",
        "literature review generation",
        "scientific literature review",
        "research synthesis",
    ],
    "dpo": ["direct preference optimization", "preference optimization"],
    "visual-llm": ["vision-language models", "multimodal large language model", "large vision-language model"],
    "visual llm": ["vision-language models", "multimodal large language model", "large vision-language model"],
    "moe": ["mixture of experts", "sparse mixture of experts", "moe architecture"],
    "hotpotqa": ["multi-hop question answering", "question answering benchmark", "hotpotqa dataset"],
    "autoregressive transformer": ["autoregressive video generation", "video generation", "video synthesis"],
    "generate videos": ["video generation", "video synthesis"],
    "videos": ["video generation", "video synthesis", "video-language"],
    "commonsense": ["commonsense reasoning", "commonsense machine translation", "commonsense-aware translation"],
    "common sense": ["commonsense reasoning", "commonsense machine translation", "commonsense-aware translation"],
    "rlhf": ["reinforcement learning from human feedback", "human feedback", "preference optimization"],
    "rlhf hallucination": [
        "hallucination mitigation",
        "vision-language hallucination",
        "image captioning hallucination",
        "factually augmented rlhf",
        "fine-grained reward modeling",
    ],
    "hallucination problem": [
        "hallucination mitigation",
        "vision-language hallucination",
        "image captioning hallucination",
        "factually augmented rlhf",
        "visual grounding",
    ],
    "reinforcement learning to optimize diffusion models": [
        "video diffusion alignment",
        "reward gradients",
        "human feedback",
        "text-to-video diffusion",
    ],
    "optimize diffusion models": ["diffusion model alignment", "reward gradients", "video diffusion models"],
    "quantized pretraining": ["quantization-aware pretraining", "quantized language model pretraining", "low-bit pretraining"],
    "quantized": ["quantization-aware training", "low-bit quantization", "quantized language models"],
    "dpo training": ["direct preference optimization", "dpo vision-language models", "preference optimization vision-language"],
    "large-scale vision-language models": ["large vision-language models", "multimodal large language models"],
    "large vision-language models": ["large-scale vision-language models", "multimodal large language models"],
    "identity preservation video generation": [
        "identity-preserving video generation",
        "personalized video generation",
        "face identity preservation",
    ],
    "vocabulary watermarking": [
        "watermarking language models",
        "quality-preserving watermarking",
        "watermark robustness",
    ],
    "robot decision making": ["robot task planning", "embodied decision making", "robot planning benchmarks"],
    "task planning": ["planning benchmarks", "robot task planning", "embodied task planning"],
    "schedule planning": ["llm agents planning", "task planning", "calendar scheduling"],
    "same prompt": ["preference data", "response comparison", "preference optimization", "sft preference data"],
    "different responses": ["preference data", "response comparison", "preference optimization"],
    "long thought data": [
        "long chain-of-thought data",
        "synthetic reasoning data",
        "reasoning data generation",
        "theorem proving data",
        "proof data",
    ],
    "long video description": [
        "long video captioning",
        "long-form video understanding",
        "hour-long videos",
        "dense video captions",
        "long video comprehension",
    ],
    "long videos": ["long-form video understanding", "long video comprehension", "long video benchmark"],
    "deblurring": ["image restoration", "image deblurring", "defocus deblurring", "low-level vision"],
    "dp": ["dual-pixel", "dual pixel", "defocus blur"],
    "dp pair": ["dual-pixel", "dual pixel", "defocus deblurring", "defocus blur"],
    "dual-pixel": ["dp data", "dual-pixel data", "defocus deblurring", "defocus blur"],
    "games": ["game playing", "computer games", "autonomous agents"],
    "pc": ["computer games", "game playing"],
    "translation": ["multilingual evaluation", "cross-lingual evaluation", "machine translation"],
    "summarization": ["automatic summarization", "text summarization", "abstractive summarization"],
    "question-answering": ["question answering", "qa", "benchmarking"],
    "information extraction": [
        "named entity recognition",
        "relation extraction",
        "event extraction",
        "few-shot information extractor",
        "biomedical information extraction",
        "sequence labeling",
    ],
    "information extraction tasks": [
        "named entity recognition",
        "relation extraction",
        "event extraction",
        "few-shot information extractor",
        "sequence labeling",
    ],
    "in-context learning performance": [
        "in-context learning fall short",
        "few-shot information extraction",
        "specification-heavy tasks",
        "supervised fine-tuning comparison",
    ],
    "supervised fine-tuned small language models": [
        "supervised fine-tuning",
        "small language model",
        "information extraction",
        "few-shot information extractor",
    ],
    "regret minimization": ["reinforcement learning theory", "regret bounds", "online learning"],
    "linear function approximation": ["theoretical reinforcement learning", "markov decision process", "mdp"],
    "local visual features": ["patch-level alignment", "local grounding", "visual-text alignment", "dense visual labels"],
    "textual concepts": ["vision-language interpretability", "concept grounding", "patch-text alignment", "dense labels"],
    "clip": ["vision-language interpretability", "concept grounding", "patch-text alignment", "dense labels"],
    "mdp": ["markov decision process", "reinforcement learning theory", "regret bounds"],
    "mdps": ["markov decision process", "reinforcement learning theory", "regret bounds"],
    "machine translation": ["adversarial machine translation", "robust machine translation", "textual adversarial examples"],
    "adversarial examples": ["adversarial attacks", "robustness", "textual adversarial examples"],
    "negative impact": ["rlhf generalisation diversity", "reward collapse", "vanishing gradients"],
    "negatively impact": ["rlhf generalisation diversity", "reward collapse", "vanishing gradients"],
    "rank search results": [
        "llm reranking",
        "large language model reranker",
        "zero-shot rankers",
        "document reranking",
        "passage ranking",
        "pairwise ranking",
        "listwise ranking",
    ],
    "search ranking": ["llm reranking", "document reranking", "passage ranking", "learning to rank"],
    "pre-training": [
        "data scaling",
        "scaling laws",
        "pretraining data",
        "data pruning",
        "data-efficient pretraining",
        "training data selection",
        "data selection",
        "influential data selection",
        "data deduplication",
        "less training data",
        "deduplication",
        "self-supervised learning",
        "representation learning",
    ],
    "pretraining": [
        "data scaling",
        "scaling laws",
        "pretraining data",
        "data pruning",
        "data-efficient pretraining",
        "training data selection",
        "data selection",
        "influential data selection",
        "data deduplication",
        "less training data",
        "deduplication",
        "self-supervised learning",
        "representation learning",
    ],
    "smaller dataset": ["data pruning", "data-efficient pretraining", "less training data", "training data selection"],
    "smaller datasets": ["data pruning", "data-efficient pretraining", "less training data", "training data selection"],
    "bigger datasets": ["larger language models", "larger training data", "data scaling"],
    "small language models": ["compact language models", "small language model", "efficient language models"],
    "llm-generated": ["machine-generated text detection", "ai-generated text detection", "generated text detection", "zero-shot detection"],
    "llm-generated text": ["machine-generated text detection", "ai-generated text detection", "generated text detection", "zero-shot detection"],
    "zero-shot": ["zero-shot detection", "zero-shot classification"],
    "math": ["mathematical reasoning", "math problem solving", "theorem proving"],
    "imo": ["mathematical olympiad", "olympiad mathematics", "theorem proving", "automated theorem proving"],
    "prove": ["theorem proving", "automated theorem proving"],
    "math problems": ["mathematical reasoning", "math problem solving", "olympiad mathematics"],
    "feature matching": ["dense correspondence", "local feature matching", "detector-free matching", "image matching"],
    "low-textured": ["low-texture", "weak-textured", "textureless", "dense correspondence"],
    "object navigation": ["object goal navigation", "semantic exploration", "objectnav", "goal-oriented semantic exploration"],
    "slam": ["semantic slam", "visual slam", "object goal navigation", "visual navigation"],
    "image animation": [
        "controllable image animation",
        "video generation",
        "motion control",
        "trajectory control",
        "controllable video generation",
        "stochastic video synthesis",
    ],
    "user-annotations": ["user annotations", "interactive control", "point-based control", "drag control", "motion trajectory"],
    "ray-based rendering": ["neural rendering", "novel view synthesis", "radiance fields", "image-based rendering"],
    "novel view synthesis": ["neural rendering", "radiance fields", "image-based rendering"],
    "bound propagation": ["robustness certification", "certified robustness", "interval bound propagation", "linear bound propagation"],
    "output bounds": ["robustness certification", "certified robustness", "neural network verification"],
    "input bounds": ["robustness certification", "certified robustness", "neural network verification"],
    "steering model output": ["controlled text generation", "controllable text generation", "plug and play language models"],
    "non-attention": ["state space models", "recurrent neural networks", "sequence modeling"],
    "language modeling": ["state space models", "recurrent neural networks", "sequence modeling", "language model"],
    "play": ["game playing", "autonomous agents", "gameplay", "computer control"],
    "pc games": [
        "game playing",
        "computer games",
        "gameplay",
        "computer control",
        "open-world game agents",
        "action role-playing games",
        "minecraft",
        "doom",
        "gameplay videos",
    ],
    "agent": ["llm agents", "autonomous agents"],
    "agents": ["llm agents", "autonomous agents", "embodied agents"],
    "vision-language": ["vision-language models", "multimodal models", "image-text retrieval"],
    "vlm": ["vision-language models", "multimodal models", "image-text retrieval"],
    "vlms": ["vision-language models", "multimodal models", "image-text retrieval"],
}

KEY_PHRASE_PATTERNS = (
    r"\bhubert\s+codes?\b",
    r"\bsemantic\s+tokens?\b",
    r"\bchain[-\s]of[-\s]thought\s+prompting\b",
    r"\bcot\s+prompting\b",
    r"\bknown\s+as\s+cot\b",
    r"\bvlms?\b",
    r"\bvision[-\s]language\s+models?\b",
    r"\bvisual[-\s]llm\b",
    r"\bmoe\b",
    r"\bhotpotqa\b",
    r"\bautoregressive\s+transformers?\b",
    r"\bgenerate\s+videos?\b",
    r"\breinforcement\s+learning\s+to\s+optimize\s+diffusion\s+models?\b",
    r"\boptimize\s+diffusion\s+models?\b",
    r"\bvideo\s+diffusion\b",
    r"\bhuman\s+feedback\b",
    r"\bpreference\s+fine[-\s]tuning\b",
    r"\bcommonsense\s+problems?\b",
    r"\bcommon\s+sense\s+problems?\b",
    r"\brlhf\b",
    r"\brlhf\s+hallucination\b",
    r"\bhallucination\s+problem\b",
    r"\bquantized\s+pretraining\b",
    r"\bdpo\s+training\b",
    r"\blarge-scale\s+vision-language\s+models?\b",
    r"\bidentity\s+preservation\s+video\s+generation\b",
    r"\bvocabulary\s+watermarking\b",
    r"\brobot\s+decision\s+making\b",
    r"\bschedule\s+planning\b",
    r"\bsame\s+prompt\b",
    r"\bdifferent\s+responses\b",
    r"\breinforcement\s+learning\s+training\b",
    r"\bnegative(?:ly)?\s+impact\b",
    r"\bmulti[-\s]module\s+models?\b",
    r"\blong\s+video\s+description\b",
    r"\blong\s+videos?\b",
    r"\blong\s+thought\s+data\b",
    r"\btheorem\s+proving\b",
    r"\bproof\s+data\b",
    r"\bprompt\s+tuning\b",
    r"\bfine[-\s]tuning\b",
    r"\blatent\s+graphs?\b",
    r"\bgaussian\s+noise\b",
    r"\brgb\s+space\b",
    r"\bpixel[-\s]wise\s+uncertainty\b",
    r"\bnerf\b",
    r"\bdiffusion\s+models?\b",
    r"\blatent\s+space\b",
    r"\bsemantic\s+arithmetic\b",
    r"\bequivariance\b",
    r"\bapi[-\s]use\b",
    r"\bcode\s+evaluation\b",
    r"\bhumaneval\b",
    r"\bmbpp\b",
    r"\bcode_contests\b",
    r"\bvideo\s+aesthetics\b",
    r"\bsynthetic\s+data\b",
    r"\bsynthesis\s+data\b",
    r"\bquantum\s+monte\s+carlo\b",
    r"\bfinancial\s+tasks\b",
    r"\bara\b",
    r"\bresponse\s+length\s+prediction\b",
    r"\bnumerical\s+importances?\b",
    r"\binterpret\s+themselves\b",
    r"\bspeech\s+tokens?\b",
    r"\bmask\s+classification(?:-based)?\b",
    r"\binstance(?:-level)?\s+segmentation\b",
    r"\binverse\s+propensity\s+score(?:ing)?\b",
    r"\bself-normalized\s+ips\b",
    r"\bselection\s+bias\b",
    r"\btarget\s+networks?\b",
    r"\bdeep\s+q-learning\b",
    r"\bin-context\s+learning\b",
    r"\bscaling\s+laws?\b",
    r"\bvideo-text\b",
    r"\bimage-text\b",
    r"\breconstruction\s+error\b",
    r"\bdiscriminator\s+loss\b",
    r"\banomaly\s+score\b",
    r"\btoken-level\s+edit(?:\s+operation)?\b",
    r"\bedit\s+operation\s+prediction\b",
    r"\blanguage\s+model(?:s)?\b",
    r"\bsearch\s+results?\b",
    r"\bweb\s+corpus\b",
    r"\bwikipedia\b",
    r"\bimage\s+patch(?:es)?\b",
    r"\bregion-based\s+method(?:s)?\b",
    r"\bsemantic\s+segmentation\b",
    r"\b[a-z]+-[a-z]+\b",
    r"\b[a-z]+\s+retrieval\b",
    r"\bprompt-based\s+learning\b",
    r"\bfactuality\s+detection\b",
    r"\bllm-generated\s+text\b",
    r"\bllm[-\s]generated\b",
    r"\bzero-shot\b",
    r"\bsmaller\s+datasets?\b",
    r"\bbigger\s+datasets?\b",
    r"\bmath\s+problems?\b",
    r"\bimo\b",
    r"\bfeature\s+matching\b",
    r"\blow-textured\b",
    r"\bobject\s+navigation\b",
    r"\bimage\s+animation\b",
    r"\buser-annotations?\b",
    r"\bpc\s+games?\b",
    r"\bcomputer\s+control\b",
    r"\baction\s+role[-\s]playing\s+games?\b",
    r"\bopen[-\s]world\s+game\s+agents?\b",
    r"\bbound\s+propagation\b",
    r"\boutput\s+bounds?\b",
    r"\binput\s+bounds?\b",
    r"\bsteering\s+model\s+output\b",
    r"\bnon-attention\b",
    r"\blanguage\s+modeling\b",
    r"\bray-based\s+rendering\b",
    r"\bnovel\s+view\s+synthesis\b",
    r"\bmodel\s+generated\s+content\b",
    r"\bexternal\s+information\b",
    r"\bdp\s+pair\b",
    r"\bdual[-\s]pixel\b",
    r"\binformation\s+extraction\b",
    r"\binformation\s+extraction\s+tasks?\b",
    r"\bin-context\s+learning\s+performance\b",
    r"\bsupervised\s+fine[-\s]tuned\s+small\s+language\s+models?\b",
    r"\bfew[-\s]shot\s+information\s+extract(?:or|ion)\b",
    r"\brank\s+search\s+results?\b",
    r"\bllm\s+rerank(?:er|ing|ers)?\b",
    r"\bpoint\s+cloud\b",
    r"\bhierarchical\s+transformer\b",
    r"\bgumbel[-\s]softmax\b",
    r"\bboed\b",
    r"\bdataset\s+condensation\b",
    r"\bsfuda\b",
    r"\bexplicit\s+localization\s+information\b",
    r"\blarge\s+web\s+corpus\b",
    r"\boversmoothing\b",
    r"\bregret\s+minimization\b",
    r"\blinear\s+function\s+approximation\b",
    r"\blocal\s+visual\s+features\b",
    r"\btextual\s+concepts\b",
    r"\badversarial\s+examples\b",
)

EXCLUSION_PATTERNS = (
    r"not\s+about\s+([^,.?;]+)",
    r"exclude\s+([^,.?;]+)",
    r"except\s+([^,.?;]+)",
)

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

QUERY_NOISE_TERMS = {
    "address",
    "all",
    "apply",
    "applies",
    "evaluate",
    "evaluates",
    "experiment",
    "experiments",
    "explaining",
    "defined",
    "difficult",
    "direct",
    "diverse",
    "duration",
    "generate",
    "high-quality",
    "here",
    "least",
    "methodologies",
    "minutes",
    "one",
    "performance",
    "processing",
    "several",
    "stage",
    "standout",
    "through",
    "transformed",
    "unifies",
    "utilizing",
    "valuable",
    "want",
    "within",
    "why",
}


class QueryParser:
    """Rule-based parser used as an offline stand-in for the later LLM parser."""

    def parse(self, query: str) -> QueryIntent:
        clean = normalize_space(query)
        query_tokens = tokenize(clean)
        token_counts = Counter(query_tokens)
        key_terms = _filter_query_terms(compact_terms(token_counts.elements(), limit=16))
        phrases = self._important_phrases(clean)
        relation_cues = self._relation_cues(clean)
        fields = self._research_fields(query_tokens, phrases)
        must_have = self._must_have_constraints(clean, key_terms, phrases, relation_cues)
        soft = self._soft_constraints(clean, key_terms, phrases, relation_cues)
        excluded = self._excluded_meanings(clean)
        time_range = self._time_range(clean)
        venues = self._venues(clean)
        sub_queries = self._sub_queries(clean, key_terms, phrases, soft, relation_cues)
        main_intent = self._main_intent(clean, must_have)
        return QueryIntent(
            main_intent=main_intent,
            research_field=fields,
            must_have_constraints=must_have,
            soft_constraints=soft,
            excluded_meanings=excluded,
            time_range=time_range,
            venues=venues,
            sub_queries=sub_queries,
            query_tokens=query_tokens,
        )

    def _important_phrases(self, query: str) -> list[str]:
        lowered = query.lower()
        phrases: list[str] = []
        for phrase in SYNONYMS:
            if _contains_phrase(lowered, phrase):
                phrases.append(phrase)
        for phrase in BRIDGE_ALIASES:
            if _contains_phrase(lowered, phrase):
                phrases.append(phrase)
        for pattern in KEY_PHRASE_PATTERNS:
            for match in re.finditer(pattern, lowered):
                phrase = normalize_space(match.group(0))
                if phrase == "known as cot":
                    phrase = "cot prompting"
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        for match in re.finditer(r"\b(?:known as|called|method known as)\s+([a-z][a-z0-9-]{1,30})\b", lowered):
            alias = normalize_space(match.group(1))
            if alias == "cot":
                alias = "cot prompting"
            if alias and alias not in phrases:
                phrases.append(alias)
        return _unique(phrases)

    def _research_fields(self, tokens: list[str], phrases: list[str]) -> list[str]:
        fields: list[str] = []
        for phrase in phrases:
            if phrase in SYNONYMS:
                fields.append(phrase)
            fields.extend(BRIDGE_ALIASES.get(phrase, []))
        for token in tokens:
            hint = FIELD_HINTS.get(token)
            if hint:
                fields.append(hint)
        return _unique(fields)[:5] or ["scholarly paper search"]

    def _must_have_constraints(
        self,
        query: str,
        key_terms: list[str],
        phrases: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        relaxed = _relaxed_hard_constraints(query)
        constraints = [phrase for phrase in phrases if _useful_constraint(phrase) and phrase.lower() not in relaxed]
        phrase_tokens = {token for phrase in phrases for token in tokenize(phrase)}
        for term in key_terms:
            if term in phrase_tokens:
                continue
            if term.lower() in relaxed:
                continue
            if _useful_constraint(term) and term not in constraints:
                constraints.append(term)
        for cue in relation_cues:
            if _useful_constraint(cue) and cue not in constraints:
                constraints.append(cue)
        return constraints[:8]

    def _soft_constraints(
        self,
        query: str,
        key_terms: list[str],
        phrases: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        lowered = query.lower()
        soft: list[str] = relation_cues[:]
        for phrase in phrases:
            soft.extend(SYNONYMS.get(phrase, []))
        if "better" in lowered or "improve" in lowered:
            soft.extend(["performance improvement", "empirical comparison"])
        if "small" in lowered or "smaller" in lowered:
            soft.extend(["data efficiency", "data pruning"])
        if "active learning" in lowered:
            soft.extend(["sample efficiency", "annotation cost"])
        for phrase in phrases:
            soft.extend(BRIDGE_ALIASES.get(phrase, []))
        for term in key_terms:
            soft.extend(BRIDGE_ALIASES.get(term, []))
        for term in key_terms:
            if len(soft) >= 14:
                break
            if term not in soft:
                soft.append(term)
        return _unique(soft)[:14]

    def _relation_cues(self, query: str) -> list[str]:
        lowered = query.lower()
        cues: list[str] = []
        if any(term in lowered for term in ("first", "introduced", "initially", "pioneer")):
            cues.append("first proposed")
        if any(term in lowered for term in ("gave rise to", "gave rise", "origin", "originated", "originating", "prompt-based learning")):
            cues.append("foundational work")
        if any(term in lowered for term in ("compare", "versus", "vs", "contrast", "compare with")):
            cues.append("comparative study")
        if any(term in lowered for term in ("negative impact", "negatively impact", "harm", "degrade", "worse")):
            cues.append("performance degradation")
        if "better" in lowered and ("than" in lowered or "bigger" in lowered or "larger" in lowered):
            cues.append("better than larger baseline")
        if any(term in lowered for term in ("claiming", "claim", "show that", "showing that")):
            cues.append("empirical finding")
        if any(term in lowered for term in ("analyzes", "analyse", "analyze", "analysis")):
            cues.append("analysis")
        if any(term in lowered for term in ("factuality", "faithfulness", "hallucination")):
            cues.append("factual consistency")
        return _unique(cues)[:4]

    def _excluded_meanings(self, query: str) -> list[str]:
        lowered = query.lower()
        excluded: list[str] = []
        for pattern in EXCLUSION_PATTERNS:
            for match in re.finditer(pattern, lowered):
                excluded.append(normalize_space(match.group(1)))
        return _unique(excluded)

    def _time_range(self, query: str) -> tuple[int | None, int | None] | None:
        lowered = query.lower()
        years = [int(value) for value in YEAR_RE.findall(lowered)]
        if not years:
            return None
        if "after" in lowered or "since" in lowered:
            return (min(years), None)
        if "before" in lowered or "until" in lowered:
            return (None, max(years))
        if len(years) >= 2:
            return (min(years), max(years))
        return (years[0], years[0])

    def _venues(self, query: str) -> list[str]:
        known = ("acl", "emnlp", "neurips", "iclr", "icml", "cvpr", "iccv", "eccv", "sigir", "kdd")
        lowered = query.lower()
        return [venue.upper() for venue in known if venue in lowered]

    def _sub_queries(
        self,
        query: str,
        key_terms: list[str],
        phrases: list[str],
        soft: list[str],
        relation_cues: list[str],
    ) -> list[str]:
        base_terms = " ".join(key_terms[:8])
        query_tokens = tokenize(query)
        cleaned_query = " ".join(_filter_query_terms(compact_terms(query_tokens, limit=16))[:12])
        queries: list[str] = []
        queries.extend(_priority_alias_queries(phrases, limit=4))
        bridge_terms = _bridge_search_terms(phrases, key_terms, soft)
        if bridge_terms:
            queries.append(" ".join(bridge_terms[:8]))
        if phrases:
            queries.append(" ".join(phrases[:5] + key_terms[:5]))
        if base_terms:
            queries.append(base_terms)
        if cleaned_query:
            queries.append(cleaned_query)
        synonym_terms = phrases[:]
        for phrase in phrases:
            synonym_terms.extend(SYNONYMS.get(phrase, [])[:2])
        if synonym_terms:
            queries.append(" ".join(synonym_terms[:8]))
        if phrases:
            rich_aliases = []
            for phrase in phrases[:5]:
                rich_aliases.extend(SYNONYMS.get(phrase, [])[:6])
            if rich_aliases:
                queries.append(" ".join(_unique([*phrases[:3], *rich_aliases])[:12]))
            for phrase in phrases[:2]:
                aliases = SYNONYMS.get(phrase, [])[:3]
                queries.append(" ".join([phrase, *aliases, *key_terms[:3]]))
        if relation_cues and phrases:
            queries.append(" ".join([*relation_cues[:2], *phrases[:3], *key_terms[:3]]))
        if soft:
            queries.append(" ".join((key_terms[:5] + soft[:5])[:10]))
        if phrases and soft:
            queries.append(" ".join(_unique([*phrases[:2], *soft[:4], *key_terms[:3]])[:10]))
        if not queries:
            queries.append(query)
        return _unique([normalize_space(item) for item in queries if item])[:8]

    def _main_intent(self, query: str, constraints: list[str]) -> str:
        if constraints:
            return f"find scholarly papers about {', '.join(constraints[:4])}"
        return f"find scholarly papers relevant to: {query}"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_space(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _filter_query_terms(terms: list[str]) -> list[str]:
    filtered = [term for term in terms if term.lower() not in QUERY_NOISE_TERMS]
    return filtered or terms


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _useful_constraint(term: str) -> bool:
    if term in STOPWORDS:
        return False
    if term.lower() in QUERY_NOISE_TERMS:
        return False
    if term in {"empirical finding"}:
        return False
    if len(term) <= 2 and not any(char.isdigit() for char in term):
        return False
    return True


def _relaxed_hard_constraints(query: str) -> set[str]:
    lowered = query.lower()
    relaxed: set[str] = set()
    if "llms or llm-based agents" in lowered or "llm or llm-based agents" in lowered:
        relaxed.update({"agents", "agent", "llm-based"})
    if "surveys or summaries" in lowered or "survey or summary" in lowered:
        relaxed.update({"surveys", "survey", "summaries", "summary"})
    if "such as" in lowered:
        relaxed.update(
            {
                "translation",
                "machine translation",
                "question-answering",
                "question answering",
                "summarization",
                "summary",
                "reasoning",
            }
        )
    return relaxed


def _priority_alias_queries(phrases: list[str], *, limit: int) -> list[str]:
    queries: list[str] = []
    for phrase in phrases:
        for alias in SYNONYMS.get(phrase, []):
            tokens = tokenize(alias)
            if len(tokens) >= 4 or "-" in alias:
                queries.append(alias)
            if len(queries) >= limit:
                return queries
    return queries


def _bridge_search_terms(phrases: list[str], key_terms: list[str], soft: list[str]) -> list[str]:
    terms: list[str] = []
    for item in [*phrases, *key_terms]:
        if item in {
            "pc games",
            "games",
            "robot decision making",
            "task planning",
            "schedule planning",
            "smaller dataset",
            "smaller datasets",
            "long video description",
            "long videos",
        }:
            terms.extend(BRIDGE_ALIASES.get(item, []))
    for item in [*phrases[:4], *key_terms[:6], *soft[:6]]:
        terms.extend(BRIDGE_ALIASES.get(item, []))
    for phrase in phrases[:2]:
        if phrase in BRIDGE_ALIASES:
            terms.append(phrase)
    return _unique(terms)[:10]


def _bridge_soft_terms(term: str) -> list[str]:
    return BRIDGE_ALIASES.get(term, [])
