# 中文功能说明：引用网络扩展规划模块，从高相关候选中选择后续引用扩展种子。

from __future__ import annotations

from dataclasses import dataclass

from packages.scholar_core.models import Candidate, QueryIntent


@dataclass(frozen=True)
class CitationExpansionSeed:
    paper_id: str
    title: str
    reason: str
    priority: float


class CitationExpansionPlanner:
    """Selects high-value papers for future citation graph expansion."""

    def select_seeds(
        self,
        intent: QueryIntent,
        ranked: list[Candidate],
        *,
        max_seeds: int = 5,
    ) -> list[CitationExpansionSeed]:
        seeds: list[CitationExpansionSeed] = []
        for candidate in ranked:
            if not candidate.paper_id:
                continue
            if candidate.relevance in {"low_relevance", "weakly_relevant"}:
                continue
            if not candidate.matched_constraints and candidate.relevance != "highly_relevant":
                continue
            priority = candidate.final_score
            if candidate.relevance == "highly_relevant":
                priority += 0.3
            if set(intent.must_have_constraints).issubset(set(candidate.matched_constraints)):
                priority += 0.2
            reason = self._reason(candidate)
            seeds.append(
                CitationExpansionSeed(
                    paper_id=candidate.canonical_id or candidate.paper_id,
                    title=candidate.title,
                    reason=reason,
                    priority=round(priority, 6),
                )
            )
        seeds.sort(key=lambda item: item.priority, reverse=True)
        return seeds[:max_seeds]

    def _reason(self, candidate: Candidate) -> str:
        if candidate.relevance == "highly_relevant":
            return "high relevance result worth expanding through references and citations"
        if candidate.matched_constraints:
            return "partial constraint match worth checking neighboring papers"
        return "ranked candidate retained as a low-cost expansion fallback"
