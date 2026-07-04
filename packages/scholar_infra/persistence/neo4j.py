"""Neo4j HTTP client and graph-neighbor expansion helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


class Neo4jError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphNeighborHit:
    paper_id: str
    score: float
    support: int
    relations: tuple[str, ...]
    seed_ids: tuple[str, ...]
    concepts: tuple[str, ...] = ()


class Neo4jGraphClient:
    def __init__(self, url: str, username: str, password: str, database: str, graph_name: str) -> None:
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        self.database = database or "neo4j"
        self.graph_name = graph_name or "paper"
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {token}",
        }

    @classmethod
    def from_settings(cls, settings) -> "Neo4jGraphClient":
        return cls(
            settings.neo4j_http_url,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.neo4j_database,
            settings.neo4j_graph_name,
        )

    def ping(self) -> Any:
        return self.query_value("RETURN 1 AS ok")

    def query_rows(self, statement: str, parameters: dict[str, Any] | None = None, *, database: str | None = None) -> list[dict[str, Any]]:
        result = self.tx([{"statement": statement, "parameters": parameters or {}}], database=database)
        results = result.get("results") or []
        if not results:
            return []
        columns = results[0].get("columns") or []
        rows = []
        for item in results[0].get("data") or []:
            rows.append(dict(zip(columns, item.get("row") or [])))
        return rows

    def query_value(self, statement: str, parameters: dict[str, Any] | None = None, *, database: str | None = None) -> Any:
        rows = self.query_rows(statement, parameters, database=database)
        if not rows:
            return None
        first = rows[0]
        if not first:
            return None
        return next(iter(first.values()))

    def tx(self, statements: list[dict[str, Any]], database: str | None = None, timeout: int = 60) -> dict[str, Any]:
        db = database or self.database
        payload = json.dumps({"statements": statements}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/db/{db}/tx/commit",
            data=payload,
            method="POST",
            headers=self.headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise Neo4jError(f"Neo4j HTTP {exc.code}: {body}") from exc
        if result.get("errors"):
            raise Neo4jError(json.dumps(result["errors"], ensure_ascii=False))
        return result

    def expand_papers(
        self,
        seed_paper_ids: Iterable[str],
        *,
        max_neighbors: int = 30,
        min_concept_confidence: float = 0.65,
    ) -> list[GraphNeighborHit]:
        seed_ids = [paper_id for paper_id in dict.fromkeys(str(item).strip() for item in seed_paper_ids) if paper_id]
        if not seed_ids:
            return []
        aggregated: dict[str, dict[str, Any]] = {}
        for row in self.query_rows(_CITES_OUT_QUERY, self._query_params(seed_ids, max_neighbors)):
            _merge_neighbor(
                aggregated,
                row,
                relation="cites_out",
                weight=1.0,
            )
        for row in self.query_rows(_CITES_IN_QUERY, self._query_params(seed_ids, max_neighbors)):
            _merge_neighbor(
                aggregated,
                row,
                relation="cites_in",
                weight=1.1,
            )
        for row in self.query_rows(
            _CONCEPT_QUERY,
            self._query_params(seed_ids, max_neighbors, min_concept_confidence=min_concept_confidence),
        ):
            _merge_neighbor(
                aggregated,
                row,
                relation="concept",
                weight=0.8,
            )
        hits = [
            GraphNeighborHit(
                paper_id=paper_id,
                score=round(float(item["score"]), 6),
                support=int(item["support"]),
                relations=tuple(sorted(item["relations"])),
                seed_ids=tuple(sorted(item["seed_ids"])),
                concepts=tuple(sorted(item["concepts"])),
            )
            for paper_id, item in aggregated.items()
            if paper_id not in seed_ids
        ]
        hits.sort(key=lambda item: (item.score, item.support, item.paper_id), reverse=True)
        return hits[:max_neighbors]

    def search_concepts(
        self,
        concept_terms: Iterable[str],
        *,
        max_papers: int = 30,
        min_concept_confidence: float = 0.65,
    ) -> list[GraphNeighborHit]:
        terms = [term for term in dict.fromkeys(_normalize_concept_term(item) for item in concept_terms) if term]
        concept_ids = _concept_ids_for_terms(terms)
        if not concept_ids:
            return []
        rows = self.query_rows(
            _CONCEPT_SEARCH_QUERY,
            {
                "concept_ids": concept_ids,
                "graph_name": self.graph_name,
                "limit": max_papers,
                "min_concept_confidence": min_concept_confidence,
            },
        )
        hits: list[GraphNeighborHit] = []
        for row in rows:
            paper_id = str(row.get("paper_id") or "").strip()
            if not paper_id:
                continue
            support = int(row.get("support") or 0)
            confidence = float(row.get("confidence") or 0.0)
            concepts = tuple(str(item) for item in (row.get("concepts") or []) if str(item).strip())
            hits.append(
                GraphNeighborHit(
                    paper_id=paper_id,
                    score=round(support * 1.2 + confidence, 6),
                    support=support,
                    relations=("concept_search",),
                    seed_ids=(),
                    concepts=concepts,
                )
            )
        hits.sort(key=lambda item: (item.score, item.support, item.paper_id), reverse=True)
        return hits[:max_papers]

    def search_aliases(
        self,
        alias_terms: Iterable[str],
        *,
        max_papers: int = 30,
    ) -> list[GraphNeighborHit]:
        terms = [term for term in dict.fromkeys(_normalize_concept_term(item) for item in alias_terms) if term]
        if not terms:
            return []
        rows = self.query_rows(
            _ALIAS_SEARCH_QUERY,
            {
                "terms": terms,
                "concept_ids": _concept_ids_for_terms(terms),
                "graph_name": self.graph_name,
                "limit": max_papers,
            },
        )
        hits: list[GraphNeighborHit] = []
        for row in rows:
            paper_id = str(row.get("paper_id") or "").strip()
            if not paper_id:
                continue
            support = int(row.get("support") or 0)
            score = float(row.get("score") or 0.0)
            aliases = tuple(str(item) for item in (row.get("aliases") or []) if str(item).strip())
            relations = tuple(str(item) for item in (row.get("relations") or []) if str(item).strip())
            hits.append(
                GraphNeighborHit(
                    paper_id=paper_id,
                    score=round(score, 6),
                    support=support,
                    relations=relations or ("alias_search",),
                    seed_ids=(),
                    concepts=aliases,
                )
            )
        hits.sort(key=lambda item: (item.score, item.support, item.paper_id), reverse=True)
        return hits[:max_papers]

    def _query_params(
        self,
        seed_ids: list[str],
        max_neighbors: int,
        *,
        min_concept_confidence: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "seed_ids": seed_ids,
            "graph_name": self.graph_name,
            "limit": max_neighbors,
        }
        if min_concept_confidence is not None:
            params["min_concept_confidence"] = min_concept_confidence
        return params


_CITES_OUT_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Paper {paper_id: seed_id, graph_name: $graph_name})
MATCH (seed)-[rel:CITES {graph_name: $graph_name}]->(neighbor:Paper {graph_name: $graph_name})
WHERE neighbor.paper_id <> seed.paper_id
RETURN
  neighbor.paper_id AS paper_id,
  count(DISTINCT rel.mention_id) AS support,
  collect(DISTINCT seed.paper_id) AS seed_ids,
  ['cites_out'] AS relations,
  [] AS concepts
ORDER BY support DESC, paper_id ASC
LIMIT $limit
"""


_CITES_IN_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Paper {paper_id: seed_id, graph_name: $graph_name})
MATCH (neighbor:Paper {graph_name: $graph_name})-[rel:CITES {graph_name: $graph_name}]->(seed)
WHERE neighbor.paper_id <> seed.paper_id
RETURN
  neighbor.paper_id AS paper_id,
  count(DISTINCT rel.mention_id) AS support,
  collect(DISTINCT seed.paper_id) AS seed_ids,
  ['cites_in'] AS relations,
  [] AS concepts
ORDER BY support DESC, paper_id ASC
LIMIT $limit
"""


_CONCEPT_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Paper {paper_id: seed_id, graph_name: $graph_name})-[seed_rel:MENTIONS_CONCEPT {graph_name: $graph_name}]->(c:Concept {graph_name: $graph_name})
WHERE coalesce(seed_rel.confidence, 0) >= $min_concept_confidence
  AND coalesce(c.concept_type, '') IN ['method', 'model', 'domain', 'dataset']
MATCH (neighbor:Paper {graph_name: $graph_name})-[neighbor_rel:MENTIONS_CONCEPT {graph_name: $graph_name}]->(c)
WHERE coalesce(neighbor_rel.confidence, 0) >= $min_concept_confidence
  AND neighbor.paper_id <> seed.paper_id
RETURN
  neighbor.paper_id AS paper_id,
  count(DISTINCT c.concept_id) AS support,
  collect(DISTINCT seed.paper_id) AS seed_ids,
  ['concept'] AS relations,
  collect(DISTINCT c.name) AS concepts
ORDER BY support DESC, paper_id ASC
LIMIT $limit
"""


_CONCEPT_SEARCH_QUERY = """
MATCH (c:Concept {graph_name: $graph_name})
WHERE c.concept_id IN $concept_ids
MATCH (p:Paper {graph_name: $graph_name})-[rel:MENTIONS_CONCEPT {graph_name: $graph_name}]->(c)
WHERE coalesce(rel.confidence, 0) >= $min_concept_confidence
RETURN
  p.paper_id AS paper_id,
  count(DISTINCT c.concept_id) AS support,
  sum(coalesce(rel.confidence, 0)) AS confidence,
  collect(DISTINCT c.name) AS concepts
ORDER BY support DESC, confidence DESC, paper_id ASC
LIMIT $limit
"""


_ALIAS_SEARCH_QUERY = """
CALL {
  WITH $terms AS terms, $graph_name AS graph_name
  MATCH (a:Alias {graph_name: graph_name})
  WHERE a.normalized_name IN terms
  MATCH (a)-[:ALIAS_OF {graph_name: graph_name}]->(entity)
  MATCH (p:Paper {graph_name: graph_name})-[rel]->(entity)
  WHERE type(rel) IN ['PROPOSES', 'USES_DATASET', 'EVALUATES_ON', 'ADDRESSES', 'MENTIONS_CONCEPT']
    AND coalesce(rel.graph_name, graph_name) = graph_name
  RETURN
    p.paper_id AS paper_id,
    count(DISTINCT a.normalized_name) AS support,
    sum(CASE type(rel)
        WHEN 'PROPOSES' THEN 3.0
        WHEN 'USES_DATASET' THEN 2.6
        WHEN 'EVALUATES_ON' THEN 2.2
        WHEN 'ADDRESSES' THEN 2.0
        ELSE 1.3
    END) AS score,
    collect(DISTINCT a.name) + collect(DISTINCT entity.name) AS aliases,
    collect(DISTINCT type(rel)) AS relations
  UNION ALL
  WITH $terms AS terms, $graph_name AS graph_name
  MATCH (a:Alias {graph_name: graph_name})-[alias_rel:ALIAS_OF {graph_name: graph_name}]->(c:Concept {graph_name: graph_name})
  WHERE a.normalized_name IN terms
  MATCH (p:Paper {graph_name: graph_name})-[rel:MENTIONS_CONCEPT {graph_name: graph_name}]->(c)
  WHERE coalesce(rel.confidence, 0) >= 0.55
  RETURN
    p.paper_id AS paper_id,
    count(DISTINCT a.normalized_name) AS support,
    sum(1.8 + coalesce(alias_rel.confidence, 0) + coalesce(rel.confidence, 0)) AS score,
    collect(DISTINCT a.name) + collect(DISTINCT c.name) AS aliases,
    ['alias_to_concept_mentions'] AS relations
  UNION ALL
  WITH $concept_ids AS concept_ids, $graph_name AS graph_name
  MATCH (c:Concept {graph_name: graph_name})
  WHERE c.concept_id IN concept_ids
  MATCH (p:Paper {graph_name: graph_name})-[rel:MENTIONS_CONCEPT {graph_name: graph_name}]->(c)
  RETURN
    p.paper_id AS paper_id,
    count(DISTINCT c.concept_id) AS support,
    sum(1.2 + coalesce(rel.confidence, 0)) AS score,
    collect(DISTINCT c.name) AS aliases,
    ['concept_alias_fallback'] AS relations
}
RETURN
  paper_id,
  sum(support) AS support,
  sum(score) AS score,
  reduce(values = [], item IN collect(aliases) | values + item) AS aliases,
  reduce(values = [], item IN collect(relations) | values + item) AS relations
ORDER BY score DESC, support DESC, paper_id ASC
LIMIT $limit
"""


def _merge_neighbor(
    aggregated: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    relation: str,
    weight: float,
) -> None:
    paper_id = str(row.get("paper_id") or "").strip()
    if not paper_id:
        return
    item = aggregated.setdefault(
        paper_id,
        {"score": 0.0, "support": 0, "relations": set(), "seed_ids": set(), "concepts": set()},
    )
    support = int(row.get("support") or 0)
    item["score"] += weight * max(1, support)
    item["support"] += max(1, support)
    item["relations"].add(relation)
    item["seed_ids"].update(str(value) for value in row.get("seed_ids") or [] if str(value).strip())
    item["concepts"].update(str(value) for value in row.get("concepts") or [] if str(value).strip())


def _concept_ids_for_terms(terms: Iterable[str]) -> list[str]:
    concept_ids: list[str] = []
    for term in terms:
        for concept_type in _candidate_concept_types(term):
            concept_ids.append(_stable_hash(f"concept:{concept_type}", term))
    return list(dict.fromkeys(concept_ids))


def _candidate_concept_types(term: str) -> list[str]:
    text = _normalize_concept_term(term)
    types = [_concept_type(text)]
    for fallback in ("method", "model", "domain", "dataset", "other"):
        if fallback not in types:
            types.append(fallback)
    return types


def _concept_type(name: str) -> str:
    text = _normalize_concept_term(name)
    if "dataset" in text or "benchmark" in text:
        return "dataset"
    if any(term in text for term in ("f1", "recall", "precision", "accuracy", "auc")):
        return "metric"
    if any(term in text for term in ("transformer", "clip", "llm", "bert", "gpt", "model")):
        return "model"
    if any(term in text for term in ("method", "algorithm", "attention", "retrieval", "segmentation", "detection")):
        return "method"
    if any(term in text for term in ("vision", "language", "graph", "recommendation")):
        return "domain"
    return "other"


def _stable_hash(prefix: str, *parts: Any, length: int = 20) -> str:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def _normalize_concept_term(value: Any) -> str:
    text = str(value or "").lower().replace("-", " ")
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())
