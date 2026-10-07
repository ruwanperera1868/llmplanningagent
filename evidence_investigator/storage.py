"""SQLite persistence for runs, briefings, papers and the audit trail.

SQLite was chosen in the proposal for the prototype because it needs no server
and ships with Python. Briefings are stored as validated JSON so the web layer
can re-render them without re-running the agents; PostgreSQL remains the
future option for multi-user deployment. Only the question and public
bibliographic data are stored: no credentials or personal data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Appraisal, AuditEvent, Briefing, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    briefing_json TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    paper_id TEXT NOT NULL,
    paper_json TEXT NOT NULL,
    appraisal_json TEXT,
    PRIMARY KEY (run_id, paper_id)
);
CREATE TABLE IF NOT EXISTS audit_events (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    cycle INTEGER, agent TEXT, action TEXT, detail TEXT, at TEXT,
    PRIMARY KEY (run_id, seq)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because Flask may serve requests on other threads;
        # each write is a single short transaction, so contention is not a concern here.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def start_run(self, question: str, mode: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (question, mode, started_at) VALUES (?, ?, ?)",
                (question, mode, _now()))
        return int(cur.lastrowid)

    def save_result(self, run_id: int, briefing: Briefing, papers: list[Paper],
                    appraisals: dict[str, Appraisal], audit: list[AuditEvent]) -> None:
        # One transaction, so a crash can never leave a half-written run.
        with self._conn:
            self._conn.execute("UPDATE runs SET finished_at = ?, briefing_json = ? WHERE id = ?",
                               (_now(), briefing.model_dump_json(), run_id))
            self._conn.executemany(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?)",
                [(run_id, p.paper_id, p.model_dump_json(),
                  appraisals[p.paper_id].model_dump_json() if p.paper_id in appraisals else None)
                 for p in papers])
            self._conn.execute("DELETE FROM audit_events WHERE run_id = ?", (run_id,))
            self._conn.executemany(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(run_id, i, e.cycle, e.agent, e.action, e.detail, e.at) for i, e in enumerate(audit)])

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, question, mode, started_at, finished_at FROM runs ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    def load_run(self, run_id: int) -> dict | None:
        run = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        papers, appraisals = [], {}
        for row in self._conn.execute("SELECT * FROM papers WHERE run_id = ?", (run_id,)):
            paper = Paper.model_validate_json(row["paper_json"])
            papers.append(paper)
            if row["appraisal_json"]:
                appraisals[paper.paper_id] = Appraisal.model_validate_json(row["appraisal_json"])
        papers.sort(key=lambda p: p.relevance, reverse=True)
        audit = [AuditEvent(cycle=r["cycle"], agent=r["agent"], action=r["action"],
                            detail=r["detail"], at=r["at"])
                 for r in self._conn.execute(
                     "SELECT * FROM audit_events WHERE run_id = ? ORDER BY seq", (run_id,))]
        briefing = Briefing.model_validate_json(run["briefing_json"]) if run["briefing_json"] else None
        return {"run": dict(run), "briefing": briefing, "papers": papers,
                "appraisals": appraisals, "audit": audit}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
