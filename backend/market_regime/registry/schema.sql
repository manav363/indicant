-- Model Registry Schema
-- Track every training run with hyperparameters, data provenance, and results.
-- Stores per-ticker runs so we can compare, reproduce, and roll back.

CREATE TABLE IF NOT EXISTS training_runs (
    run_id          TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    model_type      TEXT NOT NULL,       -- 'gradient_boost', 'logistic'
    created_at      TEXT NOT NULL,       -- ISO 8601

    -- Data provenance
    data_start      TEXT NOT NULL,       -- date of earliest training sample
    data_end        TEXT NOT NULL,       -- date of latest training sample
    n_samples       INTEGER NOT NULL,
    n_features      INTEGER NOT NULL,
    horizon_days    INTEGER NOT NULL,
    label_threshold REAL NOT NULL,

    -- Hyperparameters (JSON blob — flexible across model types)
    hyperparams     TEXT NOT NULL,

    -- Out-of-sample performance (populated after backtest)
    oos_sharpe              REAL,
    oos_sortino             REAL,
    oos_max_dd              REAL,
    oos_turnover            REAL,
    cost_adjusted_sharpe    REAL,

    -- Classification metrics
    accuracy        REAL,
    precision       REAL,
    recall          REAL,

    -- Evaluation config
    evaluation_freq TEXT DEFAULT 'weekly',  -- 'weekly' or 'daily'; prevents comparing
                                            -- apples-to-oranges Sharpe values

    -- Permutation test results
    permutation_p_value     REAL,           -- estimated p-value against null
    n_permutations          INTEGER,        -- number of permutations run
    null_sharpe_mean        REAL,           -- mean of null distribution
    null_sharpe_std         REAL,           -- std of null distribution
    null_sharpe_95pct       REAL,           -- 95th percentile of null distribution

    -- Artifact reference
    model_artifact  TEXT,               -- relative path, e.g. "artifacts/abc123.joblib"
    feature_list    TEXT NOT NULL,       -- JSON array of feature names

    -- Lifecycle
    status          TEXT DEFAULT 'trained'
                            CHECK (status IN ('trained', 'evaluated', 'deployed', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_runs_ticker ON training_runs(ticker);
CREATE INDEX IF NOT EXISTS idx_runs_created ON training_runs(created_at);
