-- ================================================================
--  INSIDER THREAT DETECTION SYSTEM — MySQL Schema
--  Run ONCE to create all tables:
--  Command:  mysql -u root -p < schema.sql
-- ================================================================

CREATE DATABASE IF NOT EXISTS insider_threat
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE insider_threat;

-- ---------------------------------------------------------------
--  TABLE 1: user_sessions
--  Raw activity log. Your agents / monitoring tools write here.
--  One row = one user session event.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_sessions (
    id             BIGINT        AUTO_INCREMENT PRIMARY KEY,
    user_id        VARCHAR(50)   NOT NULL,
    session_date   DATE          NOT NULL,
    time_slot      ENUM('morning','afternoon','evening','night')
                   NOT NULL DEFAULT 'morning',
    logon_count    INT           NOT NULL DEFAULT 1,
    file_count     INT           NOT NULL DEFAULT 0,
    email_count    INT           NOT NULL DEFAULT 0,
    http_count     INT           NOT NULL DEFAULT 0,
    usb_usage      INT           NOT NULL DEFAULT 0,
    is_weekend     TINYINT(1)    NOT NULL DEFAULT 0,
    source_ip      VARCHAR(45)   DEFAULT NULL,
    department     VARCHAR(100)  DEFAULT NULL,
    created_at     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_user_date (user_id, session_date),
    INDEX idx_created   (created_at),
    INDEX idx_user_id   (user_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------
--  TABLE 2: threat_scores
--  Written by Flask every 30 seconds after model scoring.
--  One row per user per scan cycle.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS threat_scores (
    id               BIGINT         AUTO_INCREMENT PRIMARY KEY,
    user_id          VARCHAR(50)    NOT NULL,
    scan_time        TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    anomaly_score    DECIMAL(6,4)   NOT NULL DEFAULT 0.0000,
    iso_score        DECIMAL(6,4)   NOT NULL DEFAULT 0.0000,
    rc_score         DECIMAL(6,4)   NOT NULL DEFAULT 0.0000,
    dbscan_outlier   TINYINT(1)     NOT NULL DEFAULT 0,
    risk_tier        ENUM('LOW','MEDIUM','HIGH','CRITICAL')
                     NOT NULL DEFAULT 'LOW',
    is_threat        TINYINT(1)     NOT NULL DEFAULT 0,
    usb_usage_sum    INT            DEFAULT 0,
    file_count_sum   INT            DEFAULT 0,
    email_count_sum  INT            DEFAULT 0,
    night_sessions   INT            DEFAULT 0,
    evening_sessions INT            DEFAULT 0,
    off_hours_pct    DECIMAL(5,4)   DEFAULT 0.0000,
    risk_composite   INT            DEFAULT 0,
    sessions_count   INT            DEFAULT 0,

    INDEX idx_user_scan (user_id, scan_time),
    INDEX idx_threat    (is_threat, scan_time),
    INDEX idx_tier      (risk_tier),
    INDEX idx_scan_time (scan_time)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------
--  TABLE 3: alerts
--  Written when a user is HIGH or CRITICAL tier.
--  Used for the live alert feed in the dashboard.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id             BIGINT        AUTO_INCREMENT PRIMARY KEY,
    user_id        VARCHAR(50)   NOT NULL,
    alert_time     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    risk_tier      ENUM('HIGH','CRITICAL') NOT NULL,
    anomaly_score  DECIMAL(6,4)  NOT NULL,
    usb_total      INT           DEFAULT 0,
    file_total     INT           DEFAULT 0,
    night_sessions INT           DEFAULT 0,
    risk_composite INT           DEFAULT 0,
    message        TEXT          DEFAULT NULL,
    acknowledged   TINYINT(1)   DEFAULT 0,

    INDEX idx_user_alert (user_id, alert_time),
    INDEX idx_unack      (acknowledged, alert_time)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------
--  TABLE 4: scan_log
--  Audit trail — one row per scan cycle run.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scan_log (
    id             BIGINT     AUTO_INCREMENT PRIMARY KEY,
    scan_time      TIMESTAMP  DEFAULT CURRENT_TIMESTAMP,
    users_scanned  INT        DEFAULT 0,
    threats_found  INT        DEFAULT 0,
    critical_count INT        DEFAULT 0,
    high_count     INT        DEFAULT 0,
    duration_ms    INT        DEFAULT 0
) ENGINE=InnoDB;

-- ---------------------------------------------------------------
--  CONVENIENCE VIEW: latest threat per user
-- ---------------------------------------------------------------
CREATE OR REPLACE VIEW latest_threats AS
SELECT ts.user_id, ts.anomaly_score, ts.risk_tier, ts.is_threat,
       ts.usb_usage_sum, ts.file_count_sum, ts.night_sessions,
       ts.risk_composite, ts.scan_time
FROM threat_scores ts
INNER JOIN (
    SELECT user_id, MAX(scan_time) AS latest
    FROM threat_scores GROUP BY user_id
) ls ON ts.user_id = ls.user_id AND ts.scan_time = ls.latest
WHERE ts.is_threat = 1
ORDER BY ts.anomaly_score DESC;

SELECT 'Schema created successfully' AS status;
