"""Flask web interface.

The web layer only collects a question, starts an investigation and renders
stored results. It holds no agent logic, so the same agents serve the CLI,
the tests and the browser (separation of presentation from reasoning, as in
the proposal's layered architecture). Jinja2 autoescaping is relied on for
every value that came from an API or the LLM, so retrieved text cannot inject
HTML into the page.

Run with:  python -m evidence_investigator.web.app  (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from ..config import Settings
from ..investigator import Investigator
from ..rendering import in_text, reference
from ..storage import Storage


def create_app(settings: Settings | None = None, investigator_factory=None) -> Flask:
    settings = settings or Settings.from_env()
    storage = Storage(settings.db_path)
    # A factory (rather than one shared Investigator) lets the offline toggle
    # on the form choose the configuration per request, and lets tests inject fakes.
    factory = investigator_factory or (
        lambda offline: Investigator(Settings.from_env(offline=offline, db_path=settings.db_path),
                                     storage=storage))

    app = Flask(__name__)
    app.jinja_env.globals.update(in_text=in_text, reference=reference)

    @app.get("/")
    def index():
        return render_template("index.html", runs=storage.list_runs(20), error=None,
                               question="", offline=settings.offline)

    @app.post("/investigate")
    def investigate():
        question = request.form.get("question", "")
        offline = request.form.get("offline") == "on"
        try:
            # Synchronous by design for a single-user prototype; a task queue
            # would be needed before multiple users could share the service.
            result = factory(offline).run(question)
        except ValueError as exc:
            return render_template("index.html", runs=storage.list_runs(20), error=str(exc),
                                   question=question, offline=offline), 400
        return redirect(url_for("show_run", run_id=result.run_id))

    @app.get("/runs/<int:run_id>")
    def show_run(run_id: int):
        data = storage.load_run(run_id)
        if data is None or data["briefing"] is None:
            abort(404)
        return render_template("run.html", **data)

    @app.get("/runs/<int:run_id>.json")
    def run_json(run_id: int):
        data = storage.load_run(run_id)
        if data is None or data["briefing"] is None:
            abort(404)
        return jsonify({
            "run": data["run"] | {"briefing_json": None},
            "briefing": data["briefing"].model_dump(),
            "papers": [p.model_dump() for p in data["papers"]],
            "appraisals": {k: a.model_dump() for k, a in data["appraisals"].items()},
            "audit": [e.model_dump() for e in data["audit"]],
        })

    return app


if __name__ == "__main__":
    # Bound to localhost only: the prototype has no authentication.
    create_app().run(host="127.0.0.1", port=5000, debug=False)
