CREATE DATABASE IF NOT EXISTS scholar_search
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE scholar_search;

CREATE TABLE IF NOT EXISTS datasets (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(128) NOT NULL,
  split_name VARCHAR(64) NOT NULL,
  source_path VARCHAR(512) NOT NULL,
  row_count BIGINT UNSIGNED NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_dataset_split (name, split_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS papers (
  paper_id VARCHAR(128) NOT NULL,
  arxiv_id VARCHAR(64) NULL,
  title TEXT NOT NULL,
  abstract MEDIUMTEXT NULL,
  year INT NULL,
  published_time DATE NULL,
  venue VARCHAR(255) NULL,
  authors JSON NULL,
  citation_count INT NULL,
  source VARCHAR(64) NOT NULL,
  fulltext_key TEXT NULL,
  has_fulltext TINYINT(1) NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (paper_id),
  KEY idx_papers_arxiv_id (arxiv_id),
  KEY idx_papers_year (year),
  KEY idx_papers_source (source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS paper_identifiers (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  paper_id VARCHAR(128) NOT NULL,
  id_type VARCHAR(64) NOT NULL,
  id_value VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_identifier (id_type, id_value),
  KEY idx_identifier_paper (paper_id),
  CONSTRAINT fk_identifier_paper
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS paper_chunks (
  chunk_id VARCHAR(160) NOT NULL,
  paper_id VARCHAR(128) NOT NULL,
  chunk_index INT NOT NULL,
  chunk_type VARCHAR(64) NOT NULL,
  section_title TEXT NULL,
  text MEDIUMTEXT NOT NULL,
  token_estimate INT NULL,
  source VARCHAR(64) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id),
  KEY idx_chunks_paper (paper_id),
  KEY idx_chunks_type (chunk_type),
  CONSTRAINT fk_chunk_paper
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS queries (
  qid VARCHAR(128) NOT NULL,
  dataset_name VARCHAR(128) NOT NULL,
  split_name VARCHAR(64) NOT NULL,
  query_text MEDIUMTEXT NOT NULL,
  published_time DATE NULL,
  answer_count INT NOT NULL DEFAULT 0,
  source_path VARCHAR(512) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (qid),
  KEY idx_queries_dataset_split (dataset_name, split_name),
  KEY idx_queries_published_time (published_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS gold_labels (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  qid VARCHAR(128) NOT NULL,
  paper_id VARCHAR(128) NOT NULL,
  arxiv_id VARCHAR(64) NULL,
  title TEXT NULL,
  label_rank INT NOT NULL,
  source VARCHAR(64) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_gold_qid_paper (qid, paper_id),
  KEY idx_gold_qid (qid),
  KEY idx_gold_paper (paper_id),
  CONSTRAINT fk_gold_query
    FOREIGN KEY (qid) REFERENCES queries(qid)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS eval_sets (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  dataset_name VARCHAR(128) NOT NULL,
  split_name VARCHAR(64) NOT NULL,
  qid VARCHAR(128) NOT NULL,
  gold_paper_ids JSON NOT NULL,
  published_time DATE NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_qid (dataset_name, split_name, qid),
  KEY idx_eval_dataset_split (dataset_name, split_name),
  CONSTRAINT fk_eval_query
    FOREIGN KEY (qid) REFERENCES queries(qid)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS ingestion_runs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_name VARCHAR(128) NOT NULL,
  source VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL,
  config JSON NULL,
  started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP NULL,
  stats JSON NULL,
  PRIMARY KEY (id),
  KEY idx_ingestion_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS search_runs (
  run_id VARCHAR(128) NOT NULL,
  qid VARCHAR(128) NOT NULL,
  strategy VARCHAR(128) NOT NULL,
  started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP NULL,
  latency_ms INT NULL,
  api_call_count INT NOT NULL DEFAULT 0,
  token_input INT NOT NULL DEFAULT 0,
  token_output INT NOT NULL DEFAULT 0,
  config JSON NULL,
  PRIMARY KEY (run_id),
  KEY idx_search_runs_qid (qid),
  CONSTRAINT fk_search_query
    FOREIGN KEY (qid) REFERENCES queries(qid)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS search_results (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(128) NOT NULL,
  qid VARCHAR(128) NOT NULL,
  paper_id VARCHAR(128) NOT NULL,
  rank_position INT NOT NULL,
  score DOUBLE NULL,
  source VARCHAR(64) NOT NULL,
  reason TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_run_rank (run_id, rank_position),
  KEY idx_results_run (run_id),
  KEY idx_results_qid (qid),
  KEY idx_results_paper (paper_id),
  CONSTRAINT fk_result_run
    FOREIGN KEY (run_id) REFERENCES search_runs(run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS api_call_logs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(128) NULL,
  provider VARCHAR(128) NOT NULL,
  endpoint VARCHAR(512) NOT NULL,
  request_hash VARCHAR(128) NULL,
  status_code INT NULL,
  latency_ms INT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_api_run (run_id),
  KEY idx_api_provider (provider)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS cost_logs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(128) NULL,
  component VARCHAR(128) NOT NULL,
  model_name VARCHAR(128) NULL,
  token_input INT NOT NULL DEFAULT 0,
  token_output INT NOT NULL DEFAULT 0,
  api_call_count INT NOT NULL DEFAULT 0,
  elapsed_ms INT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_cost_run (run_id),
  KEY idx_cost_component (component)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

