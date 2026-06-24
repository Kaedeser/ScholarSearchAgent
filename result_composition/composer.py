from __future__ import annotations

from dataclasses import asdict

from scholar_common.models import Candidate, SearchResponse


class ResultComposer:
    def to_jsonable(self, response: SearchResponse) -> dict:
        return {
            "query": response.query,
            "parsed_query": asdict(response.parsed_query),
            "plan": asdict(response.plan),
            "coverage": asdict(response.coverage),
            "cost": response.cost,
            "papers": [self._candidate_json(candidate, index) for index, candidate in enumerate(response.papers, start=1)],
        }

    def to_markdown(self, response: SearchResponse) -> str:
        lines = [
            f"# ScholarSearch Demo Result",
            "",
            f"Query: {response.query}",
            "",
            "## Parsed Query",
            "",
            f"- Main intent: {response.parsed_query.main_intent}",
            f"- Research fields: {', '.join(response.parsed_query.research_field) or '-'}",
            f"- Must-have constraints: {', '.join(response.parsed_query.must_have_constraints) or '-'}",
            f"- Sub queries: {'; '.join(response.parsed_query.sub_queries) or '-'}",
            "",
            "## Results",
            "",
            "| Rank | Paper | Year | Relevance | Score | Sources | Evidence |",
            "| ---: | --- | ---: | --- | ---: | --- | --- |",
        ]
        for index, candidate in enumerate(response.papers, start=1):
            evidence = (candidate.snippets[0] if candidate.snippets else "").replace("|", "\\|")
            title = candidate.title.replace("|", "\\|")
            sources = ", ".join(sorted(candidate.sources))
            lines.append(
                f"| {index} | {title} | {candidate.year or ''} | {candidate.relevance} | "
                f"{candidate.final_score:.3f} | {sources} | {evidence} |"
            )
        lines.extend(
            [
                "",
                "## Coverage",
                "",
                "| Constraint | Status |",
                "| --- | --- |",
            ]
        )
        for constraint, status in response.coverage.coverage.items():
            lines.append(f"| {constraint} | {status} |")
        lines.extend(
            [
                "",
                f"Stop reason: {response.coverage.reason}",
                "",
                "## Relation Graph",
                "",
                "```mermaid",
                self.to_mermaid(response),
                "```",
                "",
                "## Cost",
                "",
                "```json",
                _json_block(response.cost),
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def to_mermaid(self, response: SearchResponse) -> str:
        lines = ['graph LR', '    Q["Original query"]']
        for idx, field in enumerate(response.parsed_query.research_field[:4], start=1):
            field_id = f"F{idx}"
            lines.append(f'    Q --> {field_id}["{_escape_mermaid(field)}"]')
            for paper_index, candidate in enumerate(response.papers[:6], start=1):
                if field.lower() in f"{candidate.title} {candidate.abstract}".lower():
                    lines.append(f'    {field_id} --> P{paper_index}["{_escape_mermaid(candidate.title[:42])}"]')
        if len(lines) == 2:
            for paper_index, candidate in enumerate(response.papers[:5], start=1):
                lines.append(f'    Q --> P{paper_index}["{_escape_mermaid(candidate.title[:42])}"]')
        return "\n".join(lines)

    def to_bibtex(self, candidates: list[Candidate]) -> str:
        entries: list[str] = []
        for candidate in candidates:
            key = (candidate.paper_id.replace("arxiv:", "arxiv") or "paper").replace(".", "")
            entries.append(
                "\n".join(
                    [
                        f"@article{{{key},",
                        f"  title = {{{candidate.title}}},",
                        f"  year = {{{candidate.year or ''}}},",
                        f"  eprint = {{{candidate.paper_id.replace('arxiv:', '')}}},",
                        "  archivePrefix = {arXiv}",
                        "}",
                    ]
                )
            )
        return "\n\n".join(entries) + ("\n" if entries else "")

    def _candidate_json(self, candidate: Candidate, rank: int) -> dict:
        return {
            "rank": rank,
            "paper_id": candidate.canonical_id or candidate.paper_id,
            "title": candidate.title,
            "year": candidate.year,
            "relevance": candidate.relevance,
            "score": round(candidate.final_score, 6),
            "matched_constraints": candidate.matched_constraints,
            "missing_constraints": candidate.missing_constraints,
            "sources": sorted(candidate.sources),
            "raw_scores": candidate.raw_scores,
            "evidence": candidate.snippets[:2],
        }


def _escape_mermaid(value: str) -> str:
    return value.replace('"', "'").replace("[", "(").replace("]", ")")


def _json_block(value: dict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
