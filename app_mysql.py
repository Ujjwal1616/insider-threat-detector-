"""
================================================================
  INSIDER THREAT DETECTION — Flask + MySQL Live Backend
  File : app_mysql.py
  Run  : python app_mysql.py

  Flow every 30 seconds:
    1. Read user_sessions from MySQL (last 30 days)
    2. Aggregate each user into 30 features
    3. Run IsolationForest + DBSCAN + Risk Composite model
    4. Write results to threat_scores table
    5. Write HIGH/CRITICAL rows to alerts table
    6. Dashboard polls /api/threats to refresh

  API Endpoints:
    GET  /api/status              server health + DB status
    GET  /api/threats             latest scores for all users
    GET  /api/alerts              recent alert records
    GET  /api/stats               KPI summary for dashboard
    GET  /api/history/<user_id>   score history for one user
    POST /api/session             insert a single new session
    POST /api/sessions/batch      insert many sessions at once
    GET  /api/scan/trigger        manually trigger a scan now
    POST /api/alert/acknowledge   mark alert as acknowledged
================================================================
"""

import time
import json
import warnings
import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

import config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ================================================================
#  DATABASE
# ================================================================

def make_engine():
    from sqlalchemy.engine import URL
    url = URL.create(
        drivername = "mysql+pymysql",
        username   = config.DB_USER,
        password   = config.DB_PASSWORD,
        host       = config.DB_HOST,
        port       = config.DB_PORT,
        database   = config.DB_NAME,
        query      = {"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

try:
    ENGINE = make_engine()
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1"))
    log.info("✅ MySQL connected")
except Exception as e:
    log.error(f"❌ MySQL failed: {e}")
    log.error("   → Edit DB_PASSWORD (and other settings) in config.py")
    ENGINE = None


def db_ok() -> bool:
    if ENGINE is None:
        return False
    try:
        with ENGINE.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ================================================================
#  LOAD MODEL
# ================================================================

log.info(f"Loading model from {config.MODEL_DIR}/ ...")
scaler       = joblib.load(f"{config.MODEL_DIR}/scaler.joblib")
pca_model    = joblib.load(f"{config.MODEL_DIR}/pca.joblib")
iso          = joblib.load(f"{config.MODEL_DIR}/isolation_forest.joblib")
X_train_pca  = joblib.load(f"{config.MODEL_DIR}/X_train_pca.joblib")

with open(f"{config.MODEL_DIR}/meta.json") as f:
    META = json.load(f)

FEATURES      = META["features"]
RC_MIN        = META["rc_min"]
RC_MAX        = META["rc_max"]
EPS           = META["eps"]
MIN_SAMPLES   = META["min_samples"]
CONTAMINATION = config.CONTAMINATION
log.info(f"✅ Model ready — {len(FEATURES)} features, contamination={CONTAMINATION}")


# ================================================================
#  AUTO-SEED (populate DB from CSV on first run if table empty)
# ================================================================

def seed_database():
    if not db_ok():
        return
    try:
        with ENGINE.connect() as c:
            count = c.execute(text("SELECT COUNT(*) FROM user_sessions")).scalar()
        if count and count > 0:
            log.info(f"  DB already has {count:,} session rows — skipping seed")
            return

        log.info("  Seeding user_sessions from CSV ...")
        df = pd.read_csv(config.SEED_CSV_PATH)
        df = df.rename(columns={"user": "user_id", "day": "session_date"})

        # limit to SEED_MAX_USERS for quick testing
        users = df["user_id"].unique()[:config.SEED_MAX_USERS]
        df    = df[df["user_id"].isin(users)]

        needed = ["user_id","session_date","time_slot","logon_count",
                  "file_count","email_count","http_count","usb_usage","is_weekend"]
        df = df[[c for c in needed if c in df.columns]]
        if "is_weekend" not in df.columns:
            df["is_weekend"] = 0

        df.to_sql("user_sessions", ENGINE, if_exists="append", index=False, chunksize=500)
        log.info(f"  ✅ Seeded {len(df):,} rows for {len(users)} users")
    except Exception as e:
        log.error(f"  Seed failed: {e}")


# ================================================================
#  FEATURE ENGINEERING
# ================================================================

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["off_hours_flag"]    = (df["time_slot"].isin(["evening","night"]) | (df["is_weekend"]==1)).astype(int)
    df["usb_file_ratio"]    = df["usb_usage"]   / (df["file_count"]   + 1)
    df["email_logon_ratio"] = df["email_count"] / (df["logon_count"]  + 1)
    df["http_logon_ratio"]  = df["http_count"]  / (df["logon_count"]  + 1)
    df["exfil_activity"]    = df["file_count"]  + df["usb_usage"]
    df["total_activity"]    = df["file_count"]  + df["email_count"] + df["http_count"] + df["usb_usage"]
    df["risk_composite"] = (
          (df["time_slot"]=="night").astype(int)   *6
        + (df["time_slot"]=="evening").astype(int) *4
        + df["is_weekend"]                         *2
        + (df["usb_usage"]>0).astype(int)          *5
        + (df["usb_usage"]>5).astype(int)          *4
        + (df["usb_usage"]>15).astype(int)         *4
        + (df["file_count"]>0).astype(int)         *3
        + (df["file_count"]>10).astype(int)        *3
        + (df["file_count"]>25).astype(int)        *3
        + (df["http_count"]>5).astype(int)         *1
        - (df["email_count"]>50).astype(int)       *2
    )
    df["high_risk_session"] = (df["risk_composite"] > 8).astype(int)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer(df)
    return df.groupby("user_id").agg(
        logon_count_sum      =("logon_count","sum"),
        logon_count_max      =("logon_count","max"),
        logon_count_mean     =("logon_count","mean"),
        file_count_sum       =("file_count","sum"),
        file_count_max       =("file_count","max"),
        file_count_mean      =("file_count","mean"),
        email_count_sum      =("email_count","sum"),
        email_count_max      =("email_count","max"),
        http_count_sum       =("http_count","sum"),
        http_count_max       =("http_count","max"),
        usb_usage_sum        =("usb_usage","sum"),
        usb_usage_max        =("usb_usage","max"),
        usb_usage_mean       =("usb_usage","mean"),
        night_sessions       =("time_slot",   lambda x:(x=="night").sum()),
        evening_sessions     =("time_slot",   lambda x:(x=="evening").sum()),
        weekend_sessions     =("is_weekend","sum"),
        off_hours_pct        =("off_hours_flag","mean"),
        usb_file_ratio_max   =("usb_file_ratio","max"),
        email_logon_ratio    =("email_logon_ratio","mean"),
        http_logon_ratio     =("http_logon_ratio","mean"),
        exfil_activity_sum   =("exfil_activity","sum"),
        exfil_activity_max   =("exfil_activity","max"),
        total_activity_sum   =("total_activity","sum"),
        total_activity_max   =("total_activity","max"),
        risk_composite_sum   =("risk_composite","sum"),
        risk_composite_max   =("risk_composite","max"),
        risk_composite_mean  =("risk_composite","mean"),
        record_count         =("logon_count","count"),
        active_days          =("session_date","nunique"),
        high_risk_session_pct=("high_risk_session","mean"),
    ).reset_index()


def tier(s: float) -> str:
    if s >= 0.75: return "CRITICAL"
    if s >= 0.55: return "HIGH"
    if s >= 0.30: return "MEDIUM"
    return "LOW"


def score(agg_df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import DBSCAN as _DB

    X   = pca_model.transform(scaler.transform(agg_df[FEATURES].values))
    iso_raw  = -iso.score_samples(X)
    iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    comb     = np.vstack([X_train_pca, X])
    db_lbl   = _DB(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(comb)
    db_score = (db_lbl[len(X_train_pca):] == -1).astype(float)

    rc_raw  = agg_df["risk_composite_sum"].values.astype(float)
    rc_norm = np.clip((rc_raw - RC_MIN) / (RC_MAX - RC_MIN + 1e-9), 0, 1)

    ens   = 0.55*iso_norm + 0.25*rc_norm + 0.20*db_score
    thr   = float(np.quantile(ens, 1.0 - CONTAMINATION))
    clean = (
        (agg_df["usb_usage_sum"]==0) & (agg_df["file_count_sum"]==0) &
        (agg_df["night_sessions"]==0) & (agg_df["evening_sessions"]==0)
    ).values

    out = agg_df[["user_id"]].copy()
    out["anomaly_score"]  = np.where(clean, np.minimum(ens,0.25), ens).round(4)
    out["iso_score"]      = iso_norm.round(4)
    out["rc_score"]       = rc_norm.round(4)
    out["dbscan_outlier"] = db_score.astype(int)
    out["risk_tier"]      = out["anomaly_score"].apply(tier)
    out["is_threat"]      = np.where(clean, 0, (ens>=thr).astype(int))
    out.loc[clean, "risk_tier"] = "LOW"

    for col in ["usb_usage_sum","file_count_sum","email_count_sum",
                "night_sessions","evening_sessions","off_hours_pct",
                "risk_composite_sum","record_count"]:
        out[col] = agg_df[col].values
    return out


# ================================================================
#  SCAN — runs every 30 seconds via APScheduler
# ================================================================

_cache   = {"results": [], "scan_time": None}
_lock    = threading.Lock()


def run_scan():
    if not db_ok():
        log.warning("Scan skipped — DB not connected")
        return

    t0       = time.time()
    now      = datetime.now()
    lookback = (now - timedelta(days=config.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    log.info(f"▶  Scan @ {now.strftime('%H:%M:%S')}")

    try:
        # 1. Pull sessions from MySQL
        with ENGINE.connect() as c:
            df = pd.read_sql(text("""
                SELECT user_id, session_date, time_slot,
                       logon_count, file_count, email_count,
                       http_count, usb_usage, is_weekend
                FROM user_sessions
                WHERE session_date >= :lb
            """), c, params={"lb": lookback})

        if df.empty:
            log.warning("  No session rows found in the lookback window")
            return

        log.info(f"  {len(df):,} rows · {df['user_id'].nunique()} users")

        # 2. Aggregate + score
        agg_df  = aggregate(df)
        res_df  = score(agg_df)

        # 3. Write threat_scores
        rows = []
        for _, r in res_df.iterrows():
            rows.append({
                "user_id":r["user_id"], "scan_time":now,
                "anomaly_score":float(r["anomaly_score"]),
                "iso_score":float(r["iso_score"]),
                "rc_score":float(r["rc_score"]),
                "dbscan_outlier":int(r["dbscan_outlier"]),
                "risk_tier":r["risk_tier"],
                "is_threat":int(r["is_threat"]),
                "usb_usage_sum":int(r["usb_usage_sum"]),
                "file_count_sum":int(r["file_count_sum"]),
                "email_count_sum":int(r["email_count_sum"]),
                "night_sessions":int(r["night_sessions"]),
                "evening_sessions":int(r["evening_sessions"]),
                "off_hours_pct":float(r["off_hours_pct"]),
                "risk_composite":int(r["risk_composite_sum"]),
                "sessions_count":int(r["record_count"]),
            })
        pd.DataFrame(rows).to_sql("threat_scores", ENGINE,
                                   if_exists="append", index=False, chunksize=500)

        # 4. Write alerts for HIGH / CRITICAL
        alr = res_df[res_df["risk_tier"].isin(config.ALERT_ON_TIERS)]
        if not alr.empty:
            arw = [{
                "user_id":r["user_id"], "alert_time":now,
                "risk_tier":r["risk_tier"],
                "anomaly_score":float(r["anomaly_score"]),
                "usb_total":int(r["usb_usage_sum"]),
                "file_total":int(r["file_count_sum"]),
                "night_sessions":int(r["night_sessions"]),
                "risk_composite":int(r["risk_composite_sum"]),
                "message":(f"{r['risk_tier']} — score {r['anomaly_score']:.4f}, "
                            f"USB {int(r['usb_usage_sum'])}, files {int(r['file_count_sum'])}"),
            } for _,r in alr.iterrows()]
            pd.DataFrame(arw).to_sql("alerts", ENGINE,
                                      if_exists="append", index=False)

        # 5. Scan log
        ms = int((time.time()-t0)*1000)
        with ENGINE.connect() as c:
            c.execute(text("""
                INSERT INTO scan_log
                  (scan_time,users_scanned,threats_found,critical_count,high_count,duration_ms)
                VALUES (:st,:us,:tf,:cc,:hc,:ms)
            """), {"st":now,"us":len(res_df),
                   "tf":int(res_df["is_threat"].sum()),
                   "cc":int((res_df["risk_tier"]=="CRITICAL").sum()),
                   "hc":int((res_df["risk_tier"]=="HIGH").sum()),"ms":ms})
            c.commit()

        # 6. Cache
        with _lock:
            _cache["results"]   = res_df.to_dict(orient="records")
            _cache["scan_time"] = now.isoformat()

        log.info(f"✅  {len(res_df)} users · "
                 f"{int(res_df['is_threat'].sum())} threats · "
                 f"CRIT={int((res_df['risk_tier']=='CRITICAL').sum())} "
                 f"HIGH={int((res_df['risk_tier']=='HIGH').sum())} · {ms}ms")

    except Exception as e:
        log.error(f"❌ Scan error: {e}", exc_info=True)


# ================================================================
#  ROUTES
# ================================================================

@app.route("/api/status")
def api_status():
    return jsonify({
        "status":       "online",
        "db_connected": db_ok(),
        "db_host":      config.DB_HOST,
        "db_name":      config.DB_NAME,
        "scan_interval":config.SCAN_INTERVAL_SECONDS,
        "last_scan":    _cache["scan_time"],
        "timestamp":    datetime.now().isoformat(),
    })


@app.route("/api/threats")
def api_threats():
    """Latest scored results — dashboard polls this every 30 s."""
    with _lock:
        results   = list(_cache["results"])
        scan_time = _cache["scan_time"]

    if not results:
        return jsonify({"message":"No scan yet — first scan runs 3 s after startup",
                        "users":[],"total":0,"threats":0})

    results.sort(key=lambda x: x.get("anomaly_score",0), reverse=True)
    threats = sum(1 for r in results if r.get("is_threat"))
    tiers   = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0}
    for r in results:
        tiers[r.get("risk_tier","LOW")] += 1

    return jsonify({"users":results,"total":len(results),
                    "threats":threats,"tier_counts":tiers,
                    "last_scan":scan_time,
                    "timestamp":datetime.now().isoformat()})


@app.route("/api/alerts")
def api_alerts():
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    try:
        with ENGINE.connect() as c:
            df = pd.read_sql(text("""
                SELECT id,user_id,alert_time,risk_tier,anomaly_score,
                       usb_total,file_total,night_sessions,risk_composite,
                       message,acknowledged
                FROM alerts
                ORDER BY alert_time DESC LIMIT :lim
            """), c, params={"lim":config.MAX_ALERTS_RETURNED})
        recs = df.to_dict(orient="records")
        for r in recs:
            if hasattr(r.get("alert_time"),"isoformat"):
                r["alert_time"] = r["alert_time"].isoformat()
        return jsonify({"alerts":recs,"total":len(recs)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/stats")
def api_stats():
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    try:
        with ENGINE.connect() as c:
            total_users    = c.execute(text("SELECT COUNT(DISTINCT user_id) FROM user_sessions")).scalar() or 0
            total_sessions = c.execute(text("SELECT COUNT(*) FROM user_sessions")).scalar() or 0
            total_scans    = c.execute(text("SELECT COUNT(*) FROM scan_log")).scalar() or 0
            unack_alerts   = c.execute(text("SELECT COUNT(*) FROM alerts WHERE acknowledged=0")).scalar() or 0
            last_scan      = c.execute(text("""
                SELECT users_scanned,threats_found,critical_count,
                       high_count,duration_ms,scan_time
                FROM scan_log ORDER BY scan_time DESC LIMIT 1
            """)).fetchone()

        ls = last_scan
        return jsonify({
            "total_users":    int(total_users),
            "total_sessions": int(total_sessions),
            "total_scans":    int(total_scans),
            "unack_alerts":   int(unack_alerts),
            "scan_interval":  config.SCAN_INTERVAL_SECONDS,
            "last_scan": {
                "users_scanned":  int(ls[0]) if ls else 0,
                "threats_found":  int(ls[1]) if ls else 0,
                "critical_count": int(ls[2]) if ls else 0,
                "high_count":     int(ls[3]) if ls else 0,
                "duration_ms":    int(ls[4]) if ls else 0,
                "scan_time":      ls[5].isoformat() if ls and ls[5] else None,
            },
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/history/<user_id>")
def api_history(user_id):
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    try:
        with ENGINE.connect() as c:
            df = pd.read_sql(text("""
                SELECT scan_time,anomaly_score,iso_score,rc_score,
                       dbscan_outlier,risk_tier,is_threat,
                       usb_usage_sum,file_count_sum,night_sessions,risk_composite
                FROM threat_scores
                WHERE user_id=:uid
                ORDER BY scan_time DESC LIMIT 48
            """), c, params={"uid":user_id})
        recs = df.to_dict(orient="records")
        for r in recs:
            if hasattr(r.get("scan_time"),"isoformat"):
                r["scan_time"] = r["scan_time"].isoformat()
        return jsonify({"user_id":user_id,"history":recs,"count":len(recs)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/session", methods=["POST"])
def api_insert_session():
    """
    Insert one new session from your monitoring agent.
    Body (JSON):
    {
      "user_id":"EMP001", "session_date":"2024-05-01",
      "time_slot":"evening", "logon_count":1,
      "file_count":45, "email_count":3,
      "http_count":0,  "usb_usage":12,
      "is_weekend":0,  "source_ip":"192.168.1.50",
      "department":"Finance"
    }
    """
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    data = request.get_json() or {}
    if "user_id" not in data:
        return jsonify({"error":"user_id required"}), 400

    row = {
        "user_id":     data["user_id"],
        "session_date":data.get("session_date", datetime.now().strftime("%Y-%m-%d")),
        "time_slot":   data.get("time_slot","morning"),
        "logon_count": int(data.get("logon_count",1)),
        "file_count":  int(data.get("file_count",0)),
        "email_count": int(data.get("email_count",0)),
        "http_count":  int(data.get("http_count",0)),
        "usb_usage":   int(data.get("usb_usage",0)),
        "is_weekend":  int(data.get("is_weekend",0)),
        "source_ip":   data.get("source_ip"),
        "department":  data.get("department"),
    }
    try:
        pd.DataFrame([row]).to_sql("user_sessions", ENGINE,
                                   if_exists="append", index=False)
        return jsonify({"status":"inserted","user_id":row["user_id"],
                        "timestamp":datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/sessions/batch", methods=["POST"])
def api_batch():
    """Insert many sessions at once. Body: { "sessions": [...] }"""
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    data     = request.get_json() or {}
    sessions = data.get("sessions",[])
    if not sessions:
        return jsonify({"error":"sessions array is empty"}), 400
    try:
        df = pd.DataFrame(sessions)
        df.to_sql("user_sessions", ENGINE,
                  if_exists="append", index=False, chunksize=500)
        return jsonify({"status":"inserted","count":len(df),
                        "timestamp":datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/scan/trigger")
def api_trigger_scan():
    """Manually trigger a scan right now."""
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({"status":"scan triggered","timestamp":datetime.now().isoformat()})


@app.route("/api/alert/acknowledge", methods=["POST"])
def api_ack_alert():
    """Mark alert as acknowledged. Body: { "alert_id": 42 }"""
    if not db_ok():
        return jsonify({"error":"DB not connected"}), 503
    aid = (request.get_json() or {}).get("alert_id")
    if not aid:
        return jsonify({"error":"alert_id required"}), 400
    try:
        with ENGINE.connect() as c:
            c.execute(text("UPDATE alerts SET acknowledged=1 WHERE id=:id"),{"id":aid})
            c.commit()
        return jsonify({"status":"acknowledged","alert_id":aid})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


# ================================================================
#  STARTUP
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*58)
    print("  INSIDER THREAT DETECTION  —  MySQL Live Backend")
    print("="*58)
    print(f"  DB        : {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
    print(f"  DB Status : {'✅ Connected' if db_ok() else '❌ Not connected — fix config.py'}")
    print(f"  Model     : IsolationForest + DBSCAN + Risk Composite")
    print(f"  Features  : {len(FEATURES)}")
    print(f"  Scan every: {config.SCAN_INTERVAL_SECONDS} seconds")
    print(f"  Endpoints :")
    print(f"    GET  /api/status          — health check")
    print(f"    GET  /api/threats         — latest threat scores")
    print(f"    GET  /api/alerts          — alert feed")
    print(f"    GET  /api/stats           — KPI summary")
    print(f"    GET  /api/scan/trigger    — manual scan now")
    print(f"    POST /api/session         — insert one session")
    print(f"    POST /api/sessions/batch  — insert many sessions")
    print("="*58 + "\n")

    # seed DB from CSV if empty
    if config.AUTO_SEED_FROM_CSV and db_ok():
        seed_database()

    # start 30-second scheduler
    sched = BackgroundScheduler()
    sched.add_job(run_scan, "interval",
                  seconds=config.SCAN_INTERVAL_SECONDS,
                  next_run_time=datetime.now() + timedelta(seconds=3))
    sched.start()
    log.info(f"Scheduler started — scan every {config.SCAN_INTERVAL_SECONDS}s")

    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=False, use_reloader=False)
