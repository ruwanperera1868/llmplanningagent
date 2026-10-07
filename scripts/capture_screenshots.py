"""Capture the web-interface screenshots in evidence/screenshots/.

Optional helper used to produce execution evidence. Needs `pip install playwright`
and a Chromium build (set CHROMIUM_PATH, or run `playwright install chromium`).
Start the app first:  INVESTIGATOR_DB=data/demo.db python -m evidence_investigator.web.app
"""

import os

from playwright.sync_api import sync_playwright

BASE = os.environ.get("APP_URL", "http://127.0.0.1:5000")
OUT = "evidence/screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
    page = browser.new_page(viewport={"width": 1100, "height": 900})

    page.goto(BASE + "/")
    page.fill("#question", "Does self-reflection reduce hallucination in language agents?")
    page.screenshot(path=f"{OUT}/01-home-form.png", full_page=True)

    page.check("input[name=offline]")
    page.click("button[type=submit]")
    page.wait_for_url("**/runs/*")
    page.screenshot(path=f"{OUT}/02-briefing.png", full_page=True)
    page.click("details summary")
    page.locator("details").screenshot(path=f"{OUT}/03-audit-trail.png")

    page.goto(BASE + "/runs/2")
    page.click("details summary")
    page.screenshot(path=f"{OUT}/04-max-cycles-briefing.png", full_page=True)

    page.goto(BASE + "/")
    page.fill("#question", "hi?")
    page.click("button[type=submit]")
    page.screenshot(path=f"{OUT}/05-validation-error.png")

    page.goto(BASE + "/")
    page.screenshot(path=f"{OUT}/06-run-history.png", full_page=True)
    browser.close()
