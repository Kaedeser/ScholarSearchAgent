# 中文功能说明：主检索流水线测试，覆盖查询解析、候选归一、评测指标和模型服务接入。

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.scholar_api.api.routes.health import health_response
from apps.backend.scholar_api.api.schemas.search import parse_search_query
from packages.scholar_core.models import Candidate
from packages.scholar_core.model_services.ports import CrawlerStrategyPrediction, QueryIntentPrediction, QueryRewritePrediction
from packages.scholar_core.normalization.normalizer import CandidateNormalizer
from packages.scholar_core.pipeline import SearchPipeline
from packages.scholar_core.planning.planner import SearchPlanner
from packages.scholar_core.query_understanding.parser import QueryParser
from packages.scholar_eval.evaluation import score_prediction
from packages.scholar_infra.retrieval_backends.retrieval import LocalCorpus


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def make_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    write_jsonl(
        processed / "papers.jsonl",
        [
            {
                "paper_id": "arxiv:1111.00001",
                "title": "Universal Image Text Representation Learning for Image Retrieval",
                "abstract": "A model for cross-modal image retrieval and visual search.",
                "year": 2020,
                "source": "pasa",
            },
            {
                "paper_id": "arxiv:2222.00002",
                "title": "A Survey of Database Query Optimizers",
                "abstract": "This survey discusses relational query processing.",
                "year": 2018,
                "source": "pasa",
            },
        ],
    )
    write_jsonl(
        processed / "paper_chunks.jsonl",
        [
            {
                "chunk_id": "arxiv:1111.00001#chunk:0",
                "paper_id": "arxiv:1111.00001",
                "text": "Title: Universal Image Text Representation Learning for Image Retrieval\nAbstract: cross-modal retrieval.",
                "section_title": None,
            },
            {
                "chunk_id": "arxiv:1111.00001#chunk:1",
                "paper_id": "arxiv:1111.00001",
                "text": "Title: Universal Image Text Representation Learning for Image Retrieval\nSection: Method\nReferenced papers: visual search.",
                "section_title": "Method",
            },
            {
                "chunk_id": "arxiv:2222.00002#chunk:0",
                "paper_id": "arxiv:2222.00002",
                "text": "Title: A Survey of Database Query Optimizers\nAbstract: relational query plans.",
                "section_title": None,
            },
        ],
    )
    write_jsonl(
        processed / "queries.jsonl",
        [
            {
                "qid": "q1",
                "query_text": "What works are related to the field of image retrieval?",
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    write_jsonl(
        processed / "eval_sets.jsonl",
        [
            {
                "qid": "q1",
                "gold_paper_ids": ["arxiv:1111.00001"],
                "split_name": "dev",
                "dataset_name": "mini",
            }
        ],
    )
    return processed


def build_pipeline(processed: Path, *, model_services=None) -> SearchPipeline:
    corpus = LocalCorpus(processed, max_chunks_per_paper=4)
    return SearchPipeline(corpus, per_query_top_k=5, model_services=model_services)


def test_query_parser_generates_constraints_and_subqueries():
    parsed = QueryParser().parse("Find image retrieval papers after 2020")
    assert "image retrieval" in parsed.must_have_constraints
    assert parsed.time_range == (2020, None)
    assert parsed.sub_queries


def test_query_parser_expands_failure_case_terms():
    parsed = QueryParser().parse("What work proposes to model speech based on HuBERT codes or semantic tokens?")

    assert "hubert codes" in parsed.must_have_constraints
    assert "semantic tokens" in parsed.must_have_constraints
    assert any("discrete speech units" in query or "speech tokens" in query for query in parsed.sub_queries)
    assert any("generative spoken language modeling" in query for query in parsed.sub_queries)


def test_query_parser_expands_auto_scholar_locator_aliases():
    seq2edit = QueryParser().parse(
        "Which work first implemented token-level edit operation prediction in Seq2Edit methods?"
    )
    anomaly = QueryParser().parse(
        "Which paper proposed improving the anomaly score for reconstruction-based techniques via reconstruction error?"
    )
    reconstruction = QueryParser().parse("Find hybrid architectures in reconstruction-based techniques")
    ips = QueryParser().parse(
        "What research works talk about using Inverse Propensity Score and Self-Normalized IPS to tackle selection bias?"
    )
    mask = QueryParser().parse("What studies introduced mask classification-based methods for instance-level segmentation?")
    rlhf = QueryParser().parse(
        "Provide papers claiming that reinforcement learning can negatively impact supervised fine-tuned LLMs."
    )

    assert any("encode tag realize" in query or "lasertagger" in query for query in seq2edit.sub_queries)
    assert any("tranad" in query or "multivariate time series anomaly detection" in query for query in anomaly.sub_queries)
    assert any("graph attention network anomaly detection" in query for query in reconstruction.sub_queries)
    assert any("debiasing learning evaluation" in query for query in ips.sub_queries)
    assert any("detr" in query or "set prediction segmentation" in query for query in mask.sub_queries)
    assert any("reward collapse" in query or "vanishing gradients" in query for query in rlhf.sub_queries)


def test_query_parser_expands_full_eval_failure_aliases():
    cot = QueryParser().parse("Which paper introduced the method known as CoT prompting?")
    vlm = QueryParser().parse(
        "Can you specify the studies about using prompt and fine-tuning techniques for adapting VLMs?"
    )
    nerf = QueryParser().parse(
        "Which works assumed Gaussian noise in RGB space for pixel-wise uncertainty in the context of NeRF?"
    )
    code_eval = QueryParser().parse(
        "Show me code evaluation datasets harder than HumanEval and MBPP but easier than code_contests."
    )

    assert "cot prompting" in cot.must_have_constraints
    assert any("chain-of-thought prompting" in query for query in cot.sub_queries)
    assert any("clip-adapter" in query or "learning to prompt" in query for query in vlm.sub_queries)
    assert any("nerf in the wild" in query or "activenerf" in query for query in nerf.sub_queries)
    assert any("livecodebench" in query or "evalplus" in query for query in code_eval.sub_queries)


def test_query_parser_bridges_broad_failures_into_retrieval_terms():
    ranking = QueryParser().parse("Give me papers about how to rank search results by the use of LLM.")
    multilingual = QueryParser().parse(
        "Could you cite the works where multilingual LLMs were evaluated on individual tasks such as Translation, Question-Answering, Summarization, and Reasoning?"
    )
    dpo_vlm_games = QueryParser().parse(
        "DPO training for large-scale vision-language models and agents to automatically play PC games."
    )
    deblurring = QueryParser().parse(
        "Which works have implemented different kernels at various regions of the DP pair for single-task based deblurring?"
    )

    assert "information retrieval" in ranking.research_field
    assert any("learning to rank" in query or "search ranking" in query for query in ranking.sub_queries)
    assert "multilingual evaluation" in multilingual.research_field
    assert any("cross-lingual evaluation" in query or "multilingual benchmarks" in query for query in multilingual.sub_queries)
    assert any("direct preference optimization" in query or "vision-language models" in query for query in dpo_vlm_games.sub_queries)
    assert any("game playing" in query or "computer games" in query for query in dpo_vlm_games.sub_queries)
    assert any("image restoration" in query or "image deblurring" in query for query in deblurring.sub_queries)
    assert any("dual-pixel" in query or "defocus deblurring" in query for query in deblurring.sub_queries)
    assert "dual-pixel" not in deblurring.must_have_constraints
    assert "different" not in deblurring.must_have_constraints
    assert "regions" not in deblurring.must_have_constraints


def test_query_parser_preserves_dataset_and_architecture_entities():
    hotpotqa = QueryParser().parse(
        "Papers that propose methods based on large language models and evaluate their performance through experiments on the HotPotQA dataset."
    )
    visual_moe = QueryParser().parse("Give me all visual-LLM models that are MoE architecture")
    autoregressive_video = QueryParser().parse("List all papers that use autoregressive transformer to generate videos.")

    assert "hotpotqa" in hotpotqa.must_have_constraints
    assert any("multi-hop question answering" in query or "hotpotqa dataset" in query for query in hotpotqa.sub_queries)
    assert "through" not in hotpotqa.must_have_constraints
    assert "performance" not in hotpotqa.must_have_constraints
    assert "experiments" not in hotpotqa.must_have_constraints
    assert "visual-llm" in visual_moe.must_have_constraints
    assert "moe" in visual_moe.must_have_constraints
    assert "all" not in visual_moe.must_have_constraints
    assert any("mixture of experts" in query or "multimodal large language model" in query for query in visual_moe.sub_queries)
    assert "generate" not in autoregressive_video.must_have_constraints
    assert any("autoregressive video generation" in query or "video generation" in query for query in autoregressive_video.sub_queries)


def test_query_parser_bridges_real_query_failure_clusters():
    commonsense_mt = QueryParser().parse("Papers on solving common sense problems in machine translation.")
    watermark = QueryParser().parse(
        "Provide papers on methods that protect the generation quality of LLMs under vocabulary watermarking settings."
    )
    identity_video = QueryParser().parse("Show me research on identity preservation video generation.")
    dpo_vlm = QueryParser().parse("DPO training for large-scale vision-language models.")
    long_thought = QueryParser().parse(
        "Search for synthetic data of large language models that automatically generate long thought data for learning."
    )
    diffusion_rl = QueryParser().parse(
        "Show me papers utilizing reinforcement learning to optimize diffusion models for video generation."
    )
    game_agents = QueryParser().parse(
        "Find papers that use large vision-language models as agents to automatically play PC games."
    )
    ie_icl = QueryParser().parse(
        "Explain why in-context learning performance cannot surpass supervised fine-tuned small language models in information extraction tasks."
    )
    llm_rank = QueryParser().parse("Give me papers about how to rank search results by the use of LLM.")

    assert any("commonsense machine translation" in query for query in commonsense_mt.sub_queries)
    assert any("quality-preserving watermarking" in query or "watermark robustness" in query for query in watermark.sub_queries)
    assert any("identity-preserving video generation" in query for query in identity_video.sub_queries)
    assert any("dpo vision-language models" in query or "preference optimization vision-language" in query for query in dpo_vlm.sub_queries)
    assert any(
        "long chain-of-thought data" in query or "theorem proving data" in query for query in long_thought.sub_queries
    )
    assert not any(query == "reinforcement learning from human feedback" for query in diffusion_rl.sub_queries[:2])
    assert any("video diffusion alignment" in query or "reward gradients" in query for query in diffusion_rl.sub_queries)
    assert any("computer control" in query or "gameplay videos" in query for query in game_agents.sub_queries)
    assert any("few-shot information extractor" in query or "sequence labeling" in query for query in ie_icl.sub_queries)
    assert any("llm reranking" in query or "document reranking" in query for query in llm_rank.sub_queries)


def test_query_parser_handles_relational_and_foundational_phrases():
    prompt_learning = QueryParser().parse("What studies gave rise to prompt-based learning in LLMs?")
    factuality = QueryParser().parse("Which works use consistency between model generated content and external information for factuality detection in LLM?")
    web_corpus = QueryParser().parse("What studies compare using a large web corpus versus Wikipedia?")
    clip_alignment = QueryParser().parse("Could you provide me the research where local visual features aligned with textual concepts in CLIP were revealed?")
    regret = QueryParser().parse("Which works have been established for regret minimization in two types of MDPs under linear function approximation?")

    assert "foundational work" in prompt_learning.soft_constraints
    assert any("prompt learning" in query or "in-context learning" in query for query in prompt_learning.sub_queries)
    assert any("few-shot learning" in query or "pre-trained language models" in query for query in prompt_learning.sub_queries)
    assert "prompt engineering" not in prompt_learning.must_have_constraints
    assert "factual consistency" in factuality.soft_constraints
    assert any("hallucination detection" in query or "fact verification" in query for query in factuality.sub_queries)
    assert any("fact checking" in query or "factual correction" in query for query in factuality.sub_queries)
    assert "fact checking" not in factuality.must_have_constraints
    assert "comparative study" in web_corpus.soft_constraints
    assert any("pretraining corpus" in query or "common crawl" in query for query in web_corpus.sub_queries)
    assert any("patch-level alignment" in query or "concept grounding" in query for query in clip_alignment.sub_queries)
    assert any("dense labels" in query or "dense visual labels" in query for query in clip_alignment.sub_queries)
    assert "dense labels" not in clip_alignment.must_have_constraints
    assert any("regret bounds" in query or "markov decision process" in query for query in regret.sub_queries)


def test_query_parser_keeps_bridge_aliases_out_of_hard_constraints():
    parsed = QueryParser().parse("Can a smaller dataset be better than bigger datasets in LLM pretraining?")

    assert "pretraining" in parsed.must_have_constraints
    assert "better than larger baseline" in parsed.must_have_constraints
    assert any("scaling laws" in query or "pretraining data" in query for query in parsed.sub_queries)
    assert any("data pruning" in query or "data-efficient pretraining" in query for query in parsed.sub_queries)
    assert "scaling laws" not in parsed.must_have_constraints
    assert "pretraining data" not in parsed.must_have_constraints
    assert "smaller" not in parsed.must_have_constraints
    assert "bigger" not in parsed.must_have_constraints


def test_query_parser_bridges_stage4_topic_clusters():
    survey = QueryParser().parse(
        "Find papers that use LLMs or LLM-based agents to automatically write surveys or summaries for multiple scholarly documents."
    )
    generated_text = QueryParser().parse("Can LLMs detect LLM-generated text in a zero-shot manner?")
    math = QueryParser().parse("Research on teaching llms to do math prove and solve IMO level math problems.")
    navigation = QueryParser().parse("What are examples of SLAM methods in the context of object navigation tasks?")
    rendering = QueryParser().parse("Which works focused on ray-based rendering for novel view synthesis approach?")

    assert any("scientific literature review" in query or "research synthesis" in query for query in survey.sub_queries)
    assert "agents" not in survey.must_have_constraints
    assert "summaries" not in survey.must_have_constraints
    assert any("machine-generated text detection" in query or "zero-shot detection" in query for query in generated_text.sub_queries)
    assert any("mathematical reasoning" in query or "theorem proving" in query for query in math.sub_queries)
    assert any("object goal navigation" in query or "semantic exploration" in query for query in navigation.sub_queries)
    assert any("neural rendering" in query or "radiance fields" in query for query in rendering.sub_queries)


def test_query_parser_relaxes_such_as_enumerations():
    parsed = QueryParser().parse(
        "Could you cite the works where multilingual LLMs were evaluated on individual tasks such as Translation, Question-Answering, Summarization, and Reasoning?"
    )

    assert "llms" in parsed.must_have_constraints
    assert "multilingual" in parsed.must_have_constraints
    assert "translation" not in parsed.must_have_constraints
    assert "summarization" not in parsed.must_have_constraints
    assert "reasoning" not in parsed.must_have_constraints
    assert any("machine translation" in query or "question answering" in query for query in parsed.sub_queries)


def test_search_planner_uses_bounded_multi_source_actions():
    parsed = QueryParser().parse("Which work introduced mask classification-based methods for instance-level segmentation?")
    plan = SearchPlanner(per_query_top_k=60).plan(parsed)

    sources = {action.source for action in plan.search_actions}
    budget = plan.budget["retrieval_budget"]

    assert {"local_title_bm25", "local_chunk_bm25", "local_tfidf", "qdrant_dense_paper", "qdrant_sparse_paper"} <= sources
    assert {"neo4j_concept", "neo4j_alias"} <= sources
    assert any(action.top_k < 60 for action in plan.search_actions)
    assert plan.budget["max_candidates_for_selector"] == 400
    assert plan.budget["query_profile"] == "auto_locator"
    assert budget["dense_top_k"] == 100
    assert budget["sparse_paper_top_k"] == 140
    assert budget["alias_enabled"] is True
    assert budget["diversity_protected"] == 8


def test_search_planner_routes_dataset_queries_to_dense_and_alias():
    parsed = QueryParser().parse("Show me benchmark papers for code generation datasets harder than HumanEval.")
    plan = SearchPlanner(per_query_top_k=60).plan(parsed)

    dense_actions = [action for action in plan.search_actions if action.source == "qdrant_dense_paper"]
    alias_actions = [action for action in plan.search_actions if action.source == "neo4j_alias"]

    assert plan.budget["query_profile"] == "dataset_or_benchmark"
    assert plan.budget["retrieval_budget"]["alias_enabled"] is True
    assert dense_actions
    assert max(action.top_k for action in dense_actions) == 100
    assert alias_actions


def test_search_planner_routes_real_multi_answer_to_chunk_and_dense_diversity():
    parsed = QueryParser().parse("Find related papers about graph representation learning and retrieval methods.")
    plan = SearchPlanner(per_query_top_k=60).plan(parsed)
    budget = plan.budget["retrieval_budget"]
    sources = {action.source for action in plan.search_actions}

    assert plan.budget["query_profile"] == "real_multi_answer"
    assert "qdrant_dense_paper" in sources
    assert "qdrant_sparse_paper" in sources
    assert "neo4j_alias" not in sources
    assert budget["chunk_queries"] > budget["title_queries"]
    assert budget["graph_expansion_neighbors"] == 45


def test_search_planner_boosts_bridge_constraint_queries():
    parsed = QueryParser().parse(
        "Which works have implemented different kernels at various regions of the DP pair for single-task based deblurring?"
    )
    plan = SearchPlanner(per_query_top_k=60).plan(parsed)

    assert any(action.source == "local_chunk_bm25" and action.weight > 1.0 for action in plan.search_actions)
    assert any(action.source == "local_tfidf" and action.weight > 1.0 for action in plan.search_actions)


def test_neo4j_concept_term_extraction_keeps_domain_phrases():
    from packages.scholar_infra.retrieval_backends.retrieval import _concept_search_terms

    terms = _concept_search_terms(
        "mathematical reasoning | theorem proving | object goal navigation | response length prediction"
    )

    assert "mathematical reasoning" in terms
    assert "theorem proving" in terms
    assert "object goal navigation" in terms
    assert "response length prediction" in terms


def test_neo4j_alias_term_extraction_keeps_method_phrases():
    from packages.scholar_infra.retrieval_backends.retrieval import _alias_search_terms

    terms = _alias_search_terms("zero-shot detection of LLM-generated text | object goal navigation")

    assert "zero shot detection" in terms
    assert "object goal navigation" in terms


def test_neo4j_alias_terms_reuse_weighted_query_aliases():
    from packages.scholar_infra.retrieval_backends.retrieval import _alias_search_terms

    terms = _alias_search_terms("CoT prompting for VLMs")

    assert "chain of thought prompting" in terms
    assert "vision language models" in terms


def test_neo4j_alias_query_has_alias_to_concept_mentions_fallback():
    from packages.scholar_infra.persistence.neo4j import _ALIAS_SEARCH_QUERY

    assert "alias_to_concept_mentions" in _ALIAS_SEARCH_QUERY
    assert "MATCH (a:Alias" in _ALIAS_SEARCH_QUERY
    assert "MENTIONS_CONCEPT" in _ALIAS_SEARCH_QUERY


def test_alias_build_helpers_create_paper_concept_support():
    from packages.scholar_ingest.jobs.build_neo4j_graph import concept_from_alias, paper_concept_from_alias

    alias = {
        "name": "chain of thought prompting",
        "normalized_name": "chain of thought prompting",
        "target_concept_id": "concept:method:cot",
        "alias_type": "method",
        "confidence": 0.86,
        "source": "title_abstract_phrase",
    }

    concept = concept_from_alias(alias)
    edge = paper_concept_from_alias("arxiv:2201.00001", {"title": "CoT", "abstract": "Reasoning."}, alias)

    assert concept["concept_id"] == "concept:method:cot"
    assert concept["concept_type"] == "method"
    assert edge["paper_id"] == "arxiv:2201.00001"
    assert edge["concept_id"] == "concept:method:cot"
    assert edge["extractor"] == "alias_graph_quality"


def test_candidate_normalizer_merges_arxiv_aliases():
    candidates = [
        Candidate("arxiv:2301.00001", "Paper A", sources={"local_title_bm25"}, raw_scores={"a": 1.0}),
        Candidate("arXiv:2301.00001", "Paper A", sources={"local_tfidf"}, raw_scores={"b": 2.0}),
    ]
    merged = CandidateNormalizer().merge(candidates)
    assert len(merged) == 1
    assert merged[0].sources == {"local_title_bm25", "local_tfidf"}
    assert merged[0].raw_scores["b"] == 2.0


def test_score_prediction_computes_basic_metrics():
    metrics = score_prediction(["a", "b", "c"], ["b", "d"], k=3)
    assert metrics.hits == 1
    assert round(metrics.recall_at_k, 3) == 0.5
    assert round(metrics.mrr, 3) == 0.5


def test_ranker_matches_constraints_by_token_not_substring():
    pipeline_query = QueryParser().parse("Find papers about semantic tokens")
    candidate = Candidate(
        "arxiv:3333.00003",
        "Conditional Generative Fields",
        abstract="A conditional model for rendering.",
        sources={"local_title_bm25"},
        raw_scores={"local_title_bm25": 1.0},
    )
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank([candidate], pipeline_query, top_k=1)

    assert "semantic tokens" in ranked[0].missing_constraints
    assert not ranked[0].matched_constraints


def test_ranker_prefers_exact_constraint_coverage():
    parsed = QueryParser().parse("Find mask classification methods for instance-level segmentation")
    candidates = [
        Candidate(
            "arxiv:partial",
            "Per-Pixel Classification is Not All You Need for Semantic Segmentation",
            abstract="A segmentation method with classification labels.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 4.0},
        ),
        Candidate(
            "arxiv:exact",
            "Mask Classification for Instance-Level Segmentation",
            abstract="A mask classification method for instance-level segmentation.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 2.0},
        ),
    ]
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank(candidates, parsed, top_k=2)

    assert ranked[0].paper_id == "arxiv:exact"


def test_ranker_uses_soft_alias_bonus_for_locator_failures():
    parsed = QueryParser().parse(
        "Which paper proposed improving the anomaly score for reconstruction-based techniques via reconstruction error?"
    )
    candidates = [
        Candidate(
            "arxiv:generic",
            "A Survey of Anomaly Detection",
            abstract="Anomaly detection methods for industrial data.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 40.0},
        ),
        Candidate(
            "arxiv:tranad",
            "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data",
            abstract="A transformer network for anomaly detection.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 40.0},
        ),
    ]
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank(candidates, parsed, top_k=2)

    assert ranked[0].paper_id == "arxiv:tranad"
    assert ranked[0].metadata["soft_alias_bonus"] > 0


def test_ranker_uses_graph_alias_support_bonus():
    parsed = QueryParser().parse("Which paper introduced CoT prompting?")
    candidates = [
        Candidate(
            "arxiv:generic",
            "Prompting Methods for Language Models",
            abstract="A broad survey of prompting methods.",
            sources={"local_title_bm25"},
            raw_scores={"local_title_bm25": 4.0},
        ),
        Candidate(
            "arxiv:cot",
            "Chain of Thought Prompting Elicits Reasoning in Large Language Models",
            abstract="Chain of thought prompting improves reasoning.",
            sources={"neo4j_alias"},
            raw_scores={"neo4j_alias": 1.0},
            metadata={
                "alias_support": 3,
                "alias_relations": ["alias_to_concept_mentions"],
                "alias_matched_terms": ["chain of thought prompting"],
                "alias_to_concept": True,
            },
        ),
    ]
    from packages.scholar_core.ranking.ranker import CandidateRanker

    ranked = CandidateRanker().rank(candidates, parsed, top_k=2)

    assert ranked[0].paper_id == "arxiv:cot"
    assert ranked[0].metadata["graph_alias_bonus"] > 0.4


def test_source_rank_backfill_keeps_strong_source_hits_in_top50():
    parsed = QueryParser().parse("Find papers about object navigation and SLAM methods")
    from packages.scholar_core.pipeline import _source_rank_backfill

    candidates = [
        Candidate(
            f"arxiv:noise-{index}",
            f"Generic Navigation Paper {index}",
            abstract="Object navigation and learning methods.",
            sources={"local_chunk_bm25"},
            raw_scores={"local_chunk_bm25": 30.0 - index * 0.01},
            final_score=0.5 - index * 0.001,
            relevance="partially_relevant",
            metadata={"source_ranks": {"local_chunk_bm25": index + 1}},
        )
        for index in range(70)
    ]
    source_hit = Candidate(
        "arxiv:source-hit",
        "SLAM Methods for Object Goal Navigation",
        abstract="A lower scoring but high source-rank retrieval hit.",
        sources={"local_title_bm25"},
        raw_scores={"local_title_bm25": 1.0},
        final_score=0.42,
        relevance="partially_relevant",
        metadata={"source_ranks": {"local_title_bm25": 12}},
    )
    candidates.append(source_hit)

    reranked = _source_rank_backfill(candidates, parsed, top_k=50)

    top50_ids = [candidate.paper_id for candidate in reranked[:50]]
    assert "arxiv:source-hit" in top50_ids
    assert source_hit.metadata["source_rank_backfill"] == "local_title_bm25:12"


def test_source_rank_backfill_considers_wider_real_query_pool():
    parsed = QueryParser().parse("Show me papers about video diffusion models and reward gradients.")
    from packages.scholar_core.pipeline import _source_rank_backfill

    candidates = [
        Candidate(
            f"arxiv:noise-{index}",
            f"Generic Video Generation Paper {index}",
            abstract="Video generation and learning methods.",
            sources={"local_chunk_bm25"},
            raw_scores={"local_chunk_bm25": 30.0 - index * 0.01},
            final_score=0.62 - index * 0.001,
            relevance="partially_relevant",
            metadata={"source_ranks": {"local_chunk_bm25": index + 1}},
        )
        for index in range(180)
    ]
    late_source_hit = Candidate(
        "arxiv:late-source-hit",
        "Video Diffusion Alignment via Reward Gradients",
        abstract="Aligns video diffusion models with reward gradients.",
        sources={"local_title_bm25", "local_chunk_bm25"},
        raw_scores={"local_title_bm25": 1.0, "local_chunk_bm25": 1.0},
        final_score=0.43,
        relevance="partially_relevant",
        metadata={"source_ranks": {"local_title_bm25": 20, "local_chunk_bm25": 66}},
    )
    candidates.insert(160, late_source_hit)

    reranked = _source_rank_backfill(candidates, parsed, top_k=50)

    top50_ids = [candidate.paper_id for candidate in reranked[:50]]
    assert "arxiv:late-source-hit" in top50_ids


def test_candidate_preselector_limits_pool_but_keeps_source_anchors():
    parsed = QueryParser().parse("Find papers about video diffusion models and reward gradients.")
    from packages.scholar_core.ranking.preselector import CandidatePreselector

    candidates = [
        Candidate(
            f"arxiv:dense-only-{index}",
            f"Generic Video Generation Paper {index}",
            abstract="Video generation and learning methods.",
            sources={"qdrant_dense_paper"},
            raw_scores={"qdrant_dense_paper": 1.0},
            final_score=1.0 - index * 0.001,
            relevance="weakly_relevant",
            metadata={"source_ranks": {"qdrant_dense_paper": index + 1}},
        )
        for index in range(70)
    ]
    title_anchor = Candidate(
        "arxiv:title-anchor",
        "Video Diffusion Alignment via Reward Gradients",
        abstract="Aligns video diffusion models with reward gradients.",
        sources={"local_title_bm25", "local_chunk_bm25"},
        raw_scores={"local_title_bm25": 1.0, "local_chunk_bm25": 1.0},
        final_score=0.25,
        matched_constraints=["diffusion models"],
        relevance="highly_relevant",
        metadata={"source_ranks": {"local_title_bm25": 42, "local_chunk_bm25": 80}},
    )
    candidates.append(title_anchor)

    selected = CandidatePreselector().select(candidates, parsed, top_k=50, pool_limit=80)

    selected_ids = [candidate.paper_id for candidate in selected.candidates]
    assert len(selected.candidates) == 50
    assert "arxiv:title-anchor" in selected_ids
    assert title_anchor.metadata["selector_preselect_reason"] in {
        "constraint_coverage",
        "multi_source_support",
        "title_anchor",
        "score_fill",
    }
    assert selected.metadata["input_candidates"] == 71
    assert selected.metadata["selected_candidates"] == 50


def test_pipeline_returns_ranked_result(tmp_path):
    processed = make_processed_dir(tmp_path)
    pipeline = build_pipeline(processed)
    response = pipeline.search("image retrieval representation learning", top_k=2)
    assert response.papers
    assert response.papers[0].paper_id == "arxiv:1111.00001"
    assert response.plan.expand_citations_for == ["arxiv:1111.00001"]
    assert response.cost["citation_expansion_seeds"][0]["paper_id"] == "arxiv:1111.00001"
    assert response.coverage.coverage


class RecordingCorpus:
    backend_name = "recording"

    def __init__(self) -> None:
        self.actions = []

    def run_action(self, action):
        self.actions.append(action)
        return [
            Candidate(
                "arxiv:unrelated",
                "Database Query Optimizers",
                abstract="Relational query planning and database execution.",
                sources={action.source},
                raw_scores={action.source: 1.0},
            )
        ]

    def stats(self) -> dict:
        return {"backend": self.backend_name}


def test_pipeline_second_round_includes_sparse_retrieval():
    corpus = RecordingCorpus()
    pipeline = SearchPipeline(corpus, per_query_top_k=5)

    response = pipeline.search("semantic tokens for speech generation", top_k=1)

    first_round_count = len(response.plan.search_actions)
    second_round_sources = {action.source for action in corpus.actions[first_round_count:]}
    assert response.cost["rounds"] == 2
    assert "local_tfidf" in second_round_sources
    assert response.cost["diagnostic_pool_candidates"]
    assert "rrf_score" in response.cost["diagnostic_pool_candidates"][0]


def test_pipeline_graph_expansion_stays_off_for_short_locator_queries():
    class GraphCorpus(RecordingCorpus):
        def __init__(self) -> None:
            super().__init__()
            self.graph_calls = 0

        def expand_graph_candidates(self, seed_candidates, *, max_neighbors=None, min_concept_confidence=None):
            self.graph_calls += 1
            return []

    corpus = GraphCorpus()
    pipeline = SearchPipeline(corpus, per_query_top_k=5)

    pipeline.search("Which paper introduced CoT prompting?", top_k=1)

    assert corpus.graph_calls == 0


def test_pipeline_graph_expansion_can_run_for_broad_queries():
    class GraphCorpus(RecordingCorpus):
        def __init__(self) -> None:
            super().__init__()
            self.graph_calls = 0

        def expand_graph_candidates(self, seed_candidates, *, max_neighbors=None, min_concept_confidence=None):
            self.graph_calls += 1
            return [
                Candidate(
                    "arxiv:graph-neighbor",
                    "Graph Neighbor Paper",
                    abstract="A graph-related paper.",
                    sources={"neo4j"},
                    raw_scores={"neo4j_graph": 2.0},
                )
            ]

    corpus = GraphCorpus()
    pipeline = SearchPipeline(corpus, per_query_top_k=5)

    response = pipeline.search("What survey papers discuss graph representation learning and retrieval methods?", top_k=2)

    assert corpus.graph_calls >= 1
    assert any(candidate.paper_id == "arxiv:graph-neighbor" for candidate in response.papers)
    assert response.cost["model_services"].get("graph_expansion")


class FakeQueryIntentService:
    def predict_one(self, text: str) -> QueryIntentPrediction:
        return QueryIntentPrediction(
            gate_label="paper_search",
            gate_score=0.99,
            intent_label="method_search",
            intent_score=0.88,
            raw={},
        )


class FakeSelectorRerankerService:
    def rerank(self, query: str, candidates: list[Candidate], *, top_k: int):
        for candidate in candidates:
            score = 0.99 if candidate.paper_id == "arxiv:1111.00001" else 0.05
            candidate.raw_scores["selector_reranker"] = score
            candidate.final_score = score
        candidates.sort(key=lambda item: item.final_score, reverse=True)
        return candidates[:top_k], {"count": len(candidates), "threshold": 0.5}


class RecordingSelectorRerankerService(FakeSelectorRerankerService):
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def rerank(self, query: str, candidates: list[Candidate], *, top_k: int):
        self.call_sizes.append(len(candidates))
        return super().rerank(query, candidates, top_k=top_k)


class FakeCrawlerStrategyService:
    def predict(self, query: str, candidate: Candidate, *, sections: list[str]) -> CrawlerStrategyPrediction:
        return CrawlerStrategyPrediction(
            prediction="[Expand]Method[StopExpand]",
            parse_success=True,
            sections=["Method"],
            raw={},
        )


class FakeQueryRewriteService:
    def rewrite(self, text: str, *, context: dict | None = None) -> QueryRewritePrediction:
        return QueryRewritePrediction(
            rewrites=["zero-shot machine-generated text detection", "DetectGPT GECScore"],
            concepts=["machine-generated text detection", "zero-shot detection"],
            possible_answer_terms=["AI-generated text detection"],
            raw={},
        )


class FakeModelServices:
    def __init__(
        self,
        *,
        query_intent=None,
        query_rewriter=None,
        selector_reranker=None,
        crawler_strategy=None,
        selector_candidate_limit: int = 10,
        selector_pool_limit: int = 500,
        selector_protected_head: int = 0,
        crawler_top_n: int = 1,
    ) -> None:
        self.query_intent = query_intent
        self.query_rewriter = query_rewriter
        self.selector_reranker = selector_reranker
        self.crawler_strategy = crawler_strategy
        self.selector_candidate_limit = selector_candidate_limit
        self.selector_pool_limit = selector_pool_limit
        self.selector_protected_head = selector_protected_head
        self.crawler_top_n = crawler_top_n

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.query_intent:
            names.append("query_intent")
        if self.query_rewriter:
            names.append("query_rewrite")
        if self.selector_reranker:
            names.append("selector_reranker")
        if self.crawler_strategy:
            names.append("crawler_strategy")
        return names


def test_pipeline_uses_configured_model_services(tmp_path):
    processed = make_processed_dir(tmp_path)
    services = FakeModelServices(
        query_intent=FakeQueryIntentService(),
        selector_reranker=FakeSelectorRerankerService(),
        crawler_strategy=FakeCrawlerStrategyService(),
        selector_candidate_limit=10,
        crawler_top_n=1,
    )
    pipeline = build_pipeline(processed, model_services=services)

    response = pipeline.search("image retrieval representation learning", top_k=2)

    assert response.parsed_query.main_intent.startswith("method search:")
    assert response.papers[0].paper_id == "arxiv:1111.00001"
    assert response.papers[0].raw_scores["selector_reranker"] == 0.99
    assert response.papers[0].metadata["crawler_strategy"]["sections"] == ["Method"]
    assert response.cost["model_services"]["query_intent"]["intent_label"] == "method_search"
    assert response.cost["model_services"]["selector_reranker"]
    assert response.cost["model_services"]["crawler_strategy"]["papers_inspected"] == 1


def test_pipeline_preselects_500_pool_to_configured_reranker_limit():
    class ManyCandidateCorpus:
        backend_name = "many"

        def run_action(self, action):
            return [
                Candidate(
                    f"arxiv:{index:05d}",
                    f"Image Retrieval Representation Learning Candidate {index}",
                    abstract="Image retrieval representation learning and visual search.",
                    sources={action.source},
                    raw_scores={action.source: 100.0 - index},
                )
                for index in range(80)
            ]

        def stats(self) -> dict:
            return {"backend": self.backend_name}

    reranker = RecordingSelectorRerankerService()
    services = FakeModelServices(
        selector_reranker=reranker,
        selector_candidate_limit=50,
        selector_pool_limit=80,
        crawler_top_n=0,
    )
    pipeline = SearchPipeline(ManyCandidateCorpus(), per_query_top_k=80, model_services=services)

    response = pipeline.search("image retrieval representation learning", top_k=50)

    assert reranker.call_sizes == [50]
    preselector_event = response.cost["model_services"]["selector_preselector"][0]
    assert preselector_event["input_candidates"] == 80
    assert preselector_event["selected_candidates"] == 50
    reranker_event = response.cost["model_services"]["selector_reranker"][0]
    assert reranker_event["candidates"] == 50
    assert reranker_event["candidate_pool"] == 80


def test_pipeline_selector_pool_limit_is_not_inflated_by_diagnostic_top_k():
    class ManyCandidateCorpus:
        backend_name = "many"

        def run_action(self, action):
            return [
                Candidate(
                    f"arxiv:{index:05d}",
                    f"Representation Learning Candidate {index}",
                    abstract="Representation learning retrieval and visual search.",
                    sources={action.source},
                    raw_scores={action.source: 200.0 - index},
                )
                for index in range(200)
            ]

        def stats(self) -> dict:
            return {"backend": self.backend_name}

    reranker = RecordingSelectorRerankerService()
    services = FakeModelServices(
        selector_reranker=reranker,
        selector_candidate_limit=50,
        selector_pool_limit=60,
        crawler_top_n=0,
    )
    pipeline = SearchPipeline(ManyCandidateCorpus(), per_query_top_k=200, model_services=services)

    response = pipeline.search("representation learning retrieval", top_k=100)

    assert len(response.papers) == 100
    assert reranker.call_sizes == [50]
    preselector_event = response.cost["model_services"]["selector_preselector"][0]
    assert preselector_event["input_candidates"] == 60
    assert preselector_event["pool_limit"] == 60
    reranker_event = response.cost["model_services"]["selector_reranker"][0]
    assert reranker_event["candidate_pool"] == 60


def test_selector_defaults_are_fixed_to_500_pool_and_120_rerank_candidates():
    from packages.scholar_core.pipeline import _selector_candidate_limit, _selector_pool_limit, _selector_protected_head

    class MinimalModelServices:
        pass

    services = MinimalModelServices()

    assert _selector_pool_limit(services) == 500
    assert _selector_candidate_limit(services) == 120
    assert _selector_protected_head(services, top_k=50) == 0


def test_merge_reranked_head_can_protect_original_top_results():
    from packages.scholar_core.pipeline import _merge_reranked_head

    original = [
        Candidate(f"arxiv:orig-{index}", f"Original {index}", final_score=1.0 - index * 0.01)
        for index in range(5)
    ]
    reranked = [original[4], original[3], original[2], original[1], original[0]]

    merged = _merge_reranked_head(reranked, original, protected_head=2)

    assert [candidate.paper_id for candidate in merged[:5]] == [
        "arxiv:orig-0",
        "arxiv:orig-1",
        "arxiv:orig-4",
        "arxiv:orig-3",
        "arxiv:orig-2",
    ]


def test_pipeline_adds_llm_query_rewrites_without_dropping_original_actions():
    corpus = RecordingCorpus()
    services = FakeModelServices(query_rewriter=FakeQueryRewriteService(), selector_candidate_limit=10, crawler_top_n=0)
    pipeline = SearchPipeline(corpus, per_query_top_k=5, model_services=services)

    response = pipeline.search("Can LLMs detect LLM-generated text in a zero-shot manner?", top_k=2)
    action_queries = [action.query for action in corpus.actions]

    assert response.cost["model_services"]["query_rewrite"]["rewrites"]
    assert any("zero-shot machine-generated text detection" in query for query in action_queries)
    assert any("machine-generated text" in query or "llm-generated" in query for query in action_queries)


def test_pipeline_skips_llm_query_rewrite_for_low_risk_queries():
    class CountingRewriteService(FakeQueryRewriteService):
        def __init__(self) -> None:
            self.calls = 0

        def rewrite(self, text: str, *, context: dict | None = None) -> QueryRewritePrediction:
            self.calls += 1
            return super().rewrite(text, context=context)

    rewrite_service = CountingRewriteService()
    corpus = RecordingCorpus()
    services = FakeModelServices(query_rewriter=rewrite_service, selector_candidate_limit=10, crawler_top_n=0)
    pipeline = SearchPipeline(corpus, per_query_top_k=5, model_services=services)

    response = pipeline.search("graph neural network retrieval survey papers", top_k=2)

    assert rewrite_service.calls == 0
    assert response.cost["model_services"]["query_rewrite"]["skipped"] == "low_risk_query"


def test_pipeline_stops_non_paper_queries_from_query_intent(tmp_path):
    class NonPaperQueryIntentService:
        def predict_one(self, text: str) -> QueryIntentPrediction:
            return QueryIntentPrediction("non_paper_search", 0.98, None, None, {})

    processed = make_processed_dir(tmp_path)
    services = FakeModelServices(
        query_intent=NonPaperQueryIntentService(),
        selector_reranker=None,
        crawler_strategy=None,
        selector_candidate_limit=10,
        crawler_top_n=0,
    )
    pipeline = build_pipeline(processed, model_services=services)

    response = pipeline.search("Write a Python function to sort a list.", top_k=2)

    assert response.papers == []
    assert response.cost["actions_executed"] == 0
    assert response.coverage.reason == "query intent model classified the request as non-paper-search"


def test_pipeline_keeps_paper_like_short_queries_when_intent_gate_misfires():
    class NonPaperQueryIntentService:
        def predict_one(self, text: str) -> QueryIntentPrediction:
            return QueryIntentPrediction("non_paper_search", 0.98, None, None, {})

    corpus = RecordingCorpus()
    services = FakeModelServices(
        query_intent=NonPaperQueryIntentService(),
        selector_reranker=None,
        crawler_strategy=None,
        selector_candidate_limit=10,
        crawler_top_n=0,
    )
    pipeline = SearchPipeline(corpus, per_query_top_k=5, model_services=services)

    response = pipeline.search("Video aesthetics score, using multimodal large models.", top_k=2)

    assert response.cost["actions_executed"] > 0
    assert response.cost["model_services"]["query_intent"]["override"] == "paper_like_rule_fallback"
    assert response.parsed_query.main_intent.startswith("paper search fallback:")


def test_health_response_contract():
    response = health_response()
    assert response["status"] == "ok"
    assert response["service"] == "scholar-search-api"
    assert "/api/search" in response["endpoints"]


def test_search_query_schema_parses_defaults_and_top_k():
    query, top_k = parse_search_query("q=image%20retrieval&top_k=3")
    assert query == "image retrieval"
    assert top_k == 3


def test_evaluate_parser_accepts_hash_sample_order():
    from scripts.evaluate_db_agent import build_parser

    args = build_parser().parse_args(["--sample-order", "hash", "--max-queries-per-dataset", "2"])
    assert args.sample_order == "hash"
    assert args.max_queries_per_dataset == 2


def test_pool_metrics_use_diagnostic_pool_cutoffs():
    from scripts.evaluate_db_agent import build_pool_metrics

    metrics = build_pool_metrics(
        gold_ids=["gold-a", "gold-b"],
        predicted_ids=["x"] * 50 + ["gold-a"] + ["y"] * 148 + ["gold-b"],
        source_hit_ranks={
            "qdrant_dense_paper": {"gold-a": 8},
            "neo4j_alias": {"gold-b": 2},
        },
        pool_cutoffs=[50, 100, 200],
    )

    assert metrics["pool_recall"]["@50"] == 0.0
    assert metrics["pool_recall"]["@100"] == 0.5
    assert metrics["pool_recall"]["@200"] == 1.0
    assert metrics["source_gold_hits"]["qdrant_dense_paper"]["@50"] == 1
    assert metrics["source_gold_hits"]["neo4j_alias"]["@50"] == 1


def test_selector_preselection_diagnostics_are_summarized():
    from types import SimpleNamespace

    from scripts.evaluate_db_agent import summarize_selector_preselection

    summary = summarize_selector_preselection(
        [
            SimpleNamespace(
                model_events={
                    "selector_preselector": [
                        {
                            "input_candidates": 500,
                            "selected_candidates": 50,
                            "target_candidates": 50,
                            "pool_limit": 500,
                            "reason_counts": {"protected_rule_head": 10, "score_fill": 40},
                            "selected_source_counts": {"local_title_bm25": 20},
                        }
                    ]
                }
            )
        ]
    )

    assert summary["event_count"] == 1
    assert summary["query_count"] == 1
    assert summary["avg_input_candidates"] == 500.0
    assert summary["avg_selected_candidates"] == 50.0
    assert summary["avg_compression_ratio"] == 0.1
    assert summary["reason_counts"]["score_fill"] == 40
