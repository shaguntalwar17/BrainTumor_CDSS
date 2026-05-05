from __future__ import annotations

from sqlalchemy.engine import Connection, Engine


def _table_exists(conn: Connection, table_name: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name",
        {"name": table_name},
    ).first()
    return row is not None


def _column_exists(conn: Connection, table_name: str, column_name: str) -> bool:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return any(r[1] == column_name for r in rows)


def apply_lightweight_migrations(engine: Engine) -> None:
    # This project uses create_all for simplicity. These idempotent migrations
    # keep older local SQLite files compatible when new columns/tables are added.
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        if _table_exists(conn, "patients") and not _column_exists(conn, "patients", "patient_code"):
            conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN patient_code VARCHAR(64)")
            conn.exec_driver_sql("UPDATE patients SET patient_code = patient_id WHERE patient_code IS NULL")
            conn.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_patient_code ON patients (patient_code)")
        if _table_exists(conn, "patients") and not _column_exists(conn, "patients", "profile_signature"):
            conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN profile_signature VARCHAR(512)")

        if _table_exists(conn, "rag_documents") and not _column_exists(conn, "rag_documents", "document_type"):
            conn.exec_driver_sql("ALTER TABLE rag_documents ADD COLUMN document_type VARCHAR(64)")
            conn.exec_driver_sql(
                "UPDATE rag_documents SET document_type = 'scan_summary' "
                "WHERE document_type IS NULL OR document_type = ''"
            )

        if _table_exists(conn, "scans"):
            if not _column_exists(conn, "scans", "volume_manifest_path"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN volume_manifest_path VARCHAR(512)")
            if not _column_exists(conn, "scans", "corrected_mask_path"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN corrected_mask_path VARCHAR(512)")
            if not _column_exists(conn, "scans", "uncertainty_score"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN uncertainty_score FLOAT")
            if not _column_exists(conn, "scans", "uncertainty_std"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN uncertainty_std FLOAT")
            if not _column_exists(conn, "scans", "xai_method"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN xai_method VARCHAR(64)")
            if not _column_exists(conn, "scans", "correction_notes"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN correction_notes TEXT")
            if not _column_exists(conn, "scans", "stage_label"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN stage_label VARCHAR(64)")
            if not _column_exists(conn, "scans", "stage_confidence"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN stage_confidence FLOAT")
            if not _column_exists(conn, "scans", "stage_method"):
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN stage_method VARCHAR(64)")

        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS scan_probabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                class_name VARCHAR(64) NOT NULL,
                probability FLOAT DEFAULT 0.0,
                FOREIGN KEY(scan_id) REFERENCES scans(id)
            )
            """
        )
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_scan_probabilities_scan_id ON scan_probabilities (scan_id)")
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_scan_probabilities_class_name ON scan_probabilities (class_name)"
        )
