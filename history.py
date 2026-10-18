"""
src/history.py — Persistent Screening History Management
=========================================================
Manages SQLite database storage and retrieval for retinal screening records.
Ensures zero impact on ML inference and provides lightweight historical queries.
"""

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

# Default database location
DEFAULT_DB_PATH = Path("data/screening_history.db")


@contextmanager
def db_session(db_path: Path = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that opens and guarantees connection closure."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes and migrates the SQLite schema if columns are missing."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screening_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT NOT NULL,
                dr_stage INTEGER NOT NULL,
                dr_stage_name TEXT NOT NULL,
                dr_confidence REAL NOT NULL,
                image_quality TEXT NOT NULL,
                quality_passed INTEGER NOT NULL,
                uncertainty REAL NOT NULL,
                ood_score REAL NOT NULL,
                decision TEXT NOT NULL,
                metadata_json TEXT
            )
        """)
        
        # Schema migration check
        cursor.execute("PRAGMA table_info(screening_history)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "model_version" not in columns:
            cursor.execute("ALTER TABLE screening_history ADD COLUMN model_version TEXT DEFAULT 'v2.0-EfficientNetV2-S'")
        if "threshold_version" not in columns:
            cursor.execute("ALTER TABLE screening_history ADD COLUMN threshold_version TEXT DEFAULT 'calibrated-v1.4239'")


def save_screening_record(
    image_name: str,
    dr_stage: int,
    dr_stage_name: str,
    dr_confidence: float,
    image_quality: str,
    quality_passed: bool,
    uncertainty: float,
    ood_score: float,
    decision: str,
    metadata: Optional[Dict[str, Any]] = None,
    privacy_mode: bool = False,
    model_version: str = "v2.0-EfficientNetV2-S",
    threshold_version: str = "calibrated-v1.4239",
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """
    Inserts a newly completed screening analysis into the database with schema migration & privacy mode.
    
    Returns:
        The integer ID of the newly created record.
    """
    init_db(db_path)
    now_iso = datetime.now().isoformat()

    # Apply privacy mode (anonymize image filename)
    if privacy_mode:
        saved_name = f"Anonymized-Patient-{hashlib.sha256(image_name.encode()).hexdigest()[:8]}.png"
    else:
        saved_name = image_name

    meta_str = json.dumps(metadata) if metadata else "{}"

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO screening_history (
                timestamp,
                image_name,
                dr_stage,
                dr_stage_name,
                dr_confidence,
                image_quality,
                quality_passed,
                uncertainty,
                ood_score,
                decision,
                metadata_json,
                model_version,
                threshold_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso,
            saved_name,
            dr_stage,
            dr_stage_name,
            float(dr_confidence),
            str(image_quality),
            1 if quality_passed else 0,
            float(uncertainty),
            float(ood_score),
            str(decision),
            meta_str,
            model_version,
            threshold_version,
        ))
        return cursor.lastrowid


def parse_date_group(dt: datetime) -> str:
    """Categorizes a timestamp into 'Today', 'Yesterday', or 'Earlier'."""
    today = date.today()
    record_date = dt.date()
    if record_date == today:
        return "Today"
    elif record_date == today - timedelta(days=1):
        return "Yesterday"
    else:
        return "Earlier"


def get_recent_screenings(limit: int = 20, db_path: Path = DEFAULT_DB_PATH) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves the most recent screenings grouped by 'Today', 'Yesterday', and 'Earlier'.
    
    Returns:
        Dict mapping group name to list of screening records.
    """
    init_db(db_path)
    groups: Dict[str, List[Dict[str, Any]]] = {
        "Today": [],
        "Yesterday": [],
        "Earlier": [],
    }

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, image_name, dr_stage, dr_stage_name,
                   dr_confidence, image_quality, quality_passed, uncertainty,
                   ood_score, decision
            FROM screening_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

    stage_emojis = {
        0: "🟢",
        1: "🟠",
        2: "🟡",
        3: "🔴",
        4: "🟣",
    }

    for row in rows:
        d = dict(row)
        try:
            dt = datetime.fromisoformat(d["timestamp"])
            time_display = dt.strftime("%I:%M %p")
            date_display = dt.strftime("%Y-%m-%d")
            grp = parse_date_group(dt)
        except Exception:
            time_display = "--:--"
            date_display = "--"
            grp = "Earlier"

        d["time_display"] = time_display
        d["date_display"] = date_display
        d["emoji"] = stage_emojis.get(d["dr_stage"], "⚪")
        groups[grp].append(d)

    return groups


def get_screening_by_id(record_id: int, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves full details and metadata for a specific screening record."""
    init_db(db_path)
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM screening_history WHERE id = ?
        """, (record_id,))
        row = cursor.fetchone()
        if not row:
            return None

        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
        except Exception:
            d["metadata"] = {}

        try:
            dt = datetime.fromisoformat(d["timestamp"])
            d["formatted_datetime"] = dt.strftime("%Y-%m-%d %I:%M:%S %p")
            d["time_display"] = dt.strftime("%I:%M %p")
        except Exception:
            d["formatted_datetime"] = d["timestamp"]
            d["time_display"] = ""

        stage_emojis = {0: "🟢", 1: "🟠", 2: "🟡", 3: "🔴", 4: "🟣"}
        d["emoji"] = stage_emojis.get(d["dr_stage"], "⚪")
        return d


def get_all_screenings(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all screening history records for table display."""
    init_db(db_path)
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, image_name, dr_stage, dr_stage_name,
                   dr_confidence, image_quality, quality_passed, uncertainty,
                   ood_score, decision
            FROM screening_history
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()

    records = []
    for row in rows:
        d = dict(row)
        try:
            dt = datetime.fromisoformat(d["timestamp"])
            d["formatted_time"] = dt.strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            d["formatted_time"] = d["timestamp"]
        records.append(d)
    return records


def clear_all_screenings(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Deletes all records from the screening history database."""
    init_db(db_path)
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM screening_history")
