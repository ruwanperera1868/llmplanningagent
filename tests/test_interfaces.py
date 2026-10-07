"""Functional tests of the two user interfaces: the CLI and the Flask web app."""

import pytest

from conftest import FakeSource, make_paper
from evidence_investigator.cli import main
from evidence_investigator.config import Settings
from evidence_investigator.investigator import Investigator
from evidence_investigator.storage import Storage
from evidence_investigator.web.app import create_app


def test_cli_offline_writes_briefing(tmp_path, capsys):
    out = tmp_path / "briefing.md"
    code = main(["How do LLM-based agents plan, act and reflect?", "--offline", "--db",
                 str(tmp_path / "db"), "--out", str(out)])
    assert code == 0 and out.read_text().startswith("# Research briefing")
    assert "Briefing written" in capsys.readouterr().out


def test_cli_rejects_invalid_question(tmp_path, capsys):
    assert main(["hi", "--offline", "--db", str(tmp_path / "db")]) == 2
    assert "error" in capsys.readouterr().err


@pytest.fixture
def make_client(tmp_path):
    storages = []

    def build(papers):
        settings = Settings(db_path=tmp_path / "web.db", offline=True)
        storage = Storage(settings.db_path)
        storages.append(storage)
        factory = lambda offline: Investigator(settings, sources=[FakeSource([papers])],  # noqa: E731
                                               storage=storage, use_default_llm=False)
        app = create_app(settings, factory, storage)
        app.config["TESTING"] = True
        return app.test_client()

    yield build
    for s in storages:
        s.close()


def test_web_flow_question_to_briefing(make_client):
    client = make_client([make_paper("p1"), make_paper("p2", title="AI tutoring and grades")])
    assert client.get("/").status_code == 200
    resp = client.post("/investigate", data={"question": "Does generative AI improve performance?"})
    assert resp.status_code == 302
    page = client.get(resp.location)
    assert page.status_code == 200 and b"Research briefing" in page.data and b"Audit trail" in page.data
    data = client.get(resp.location + ".json").get_json()
    assert data["briefing"]["question"] == "Does generative AI improve performance?"
    assert b"Does generative AI improve performance?" in client.get("/").data  # appears in history


def test_web_rejects_bad_input_and_unknown_runs(make_client):
    client = make_client([])
    assert client.post("/investigate", data={"question": "?"}).status_code == 400
    assert client.get("/runs/42").status_code == 404


def test_web_escapes_untrusted_text_from_sources(make_client):
    """Security: a malicious title from an API must not become live HTML."""
    evil = make_paper("x", title="Generative AI <script>alert(1)</script> performance")
    client = make_client([evil])
    resp = client.post("/investigate", data={"question": "Does generative AI improve performance?"})
    page = client.get(resp.location).data
    assert b"<script>alert(1)</script>" not in page and b"&lt;script&gt;" in page
