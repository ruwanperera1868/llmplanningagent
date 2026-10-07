#!/usr/bin/env bash
# Regenerates the offline execution evidence and test report in evidence/.
set -u
cd "$(dirname "$0")/.."
rm -f data/demo.db
python -m evidence_investigator "How do LLM-based agents plan, act and reflect?" --offline -v \
  --db data/demo.db --out evidence/briefing-offline-one-cycle.md > evidence/run-offline-one-cycle.log 2>&1
python -m evidence_investigator "Does retrieval reduce hallucination?" --offline -v \
  --db data/demo.db --out evidence/briefing-offline-max-cycles.md > evidence/run-offline-max-cycles.log 2>&1
python -m evidence_investigator "hi" --offline --db data/demo.db > evidence/run-invalid-input.log 2>&1
echo "exit code: $?" >> evidence/run-invalid-input.log
python -m pytest -v --cov=evidence_investigator --cov-report=term-missing -W error::ResourceWarning \
  > evidence/test-report.txt 2>&1
echo "Evidence written to evidence/"
