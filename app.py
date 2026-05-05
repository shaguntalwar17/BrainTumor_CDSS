from __future__ import annotations

import os
from typing import Any

from flask import Flask, render_template


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    nav_items = [
        {"key": "home", "label": "Home", "href": "/", "icon": "bi-house-door-fill"},
        {"key": "upload", "label": "Upload MRI Image", "href": "/upload", "icon": "bi-cloud-arrow-up-fill"},
        {"key": "history", "label": "Patient History", "href": "/history", "icon": "bi-people-fill"},
        {"key": "comparison", "label": "Compare Reports", "href": "/comparison", "icon": "bi-clipboard2-data-fill"},
        {"key": "awareness", "label": "Brain Tumor Awareness / Prevention", "href": "/awareness", "icon": "bi-heart-pulse-fill"},
        {"key": "reports", "label": "Report Download", "href": "/reports", "icon": "bi-file-earmark-arrow-down-fill"},
    ]

    backend_api_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")
    project_name = "Brain MRI Intelligence"
    attribution = "Project made by Shagun Talwar"
    medical_disclaimer = (
        "This system is an AI-assisted academic prototype for Brain MRI tumor analysis. "
        "It is not a certified medical diagnostic tool. All predictions, segmentations, heatmaps, "
        "and reports must be verified by a certified radiologist/doctor."
    )

    def render_page(template_name: str, page_key: str, page_title: str) -> str:
        return render_template(
            template_name,
            page_title=page_title,
            current_page=page_key,
            nav_items=nav_items,
            backend_api_url=backend_api_url,
            project_name=project_name,
            attribution=attribution,
            medical_disclaimer=medical_disclaimer,
        )

    @app.route("/")
    def home() -> str:
        return render_page("home.html", "home", "Home")

    @app.route("/upload")
    def upload() -> str:
        return render_page("upload.html", "upload", "Upload MRI")

    @app.route("/history")
    def history() -> str:
        return render_page("history.html", "history", "Patient History")

    @app.route("/comparison")
    def comparison() -> str:
        return render_page("comparison.html", "comparison", "Compare Reports")

    @app.route("/awareness")
    def awareness() -> str:
        return render_page("awareness.html", "awareness", "Awareness")

    @app.route("/reports")
    def reports() -> str:
        return render_page("reports.html", "reports", "Report Download")

    @app.route("/health")
    def health() -> dict[str, Any]:
        return {"status": "healthy", "ui": "flask"}

    @app.errorhandler(404)
    def not_found(_error):
        return render_page("error.html", "home", "Page Not Found"), 404

    @app.errorhandler(500)
    def internal_error(_error):
        return render_page("error.html", "home", "Server Error"), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=False)
