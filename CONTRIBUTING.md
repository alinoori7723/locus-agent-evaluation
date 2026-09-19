# Contributing

Create a virtual environment, install the project and development tools, and run the checks:

```bash
python -m pip install -e . -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m coverage run -m pytest
python -m coverage report
```

When package source or fixtures change, regenerate the committed sample before running tests:

```bash
python -m locus_eval demo --seed 17 --output reports/sample
python -m locus_eval replay reports/sample
```

Review the resulting report and metrics. Do not change baseline expectations just to accommodate
a failing regression. A deliberate contract change should explain which observable behavior changes
and include a corresponding negative control.

Keep the runtime free of third-party dependencies unless the feature needs one. Describe tool
contracts in `docs/agent-contract.md` and threat-model changes in `docs/threat-model.md`. New fixture
families should preserve paired observations and the full nuisance cross, or explicitly introduce
and test a new construction.

Before release, build and smoke-test the wheel from a clean temporary environment:

```bash
python -m build
python scripts/check_wheel.py dist/locus_agent_evaluation-0.2.0-py3-none-any.whl
```

CI runs on Linux with Python 3.11, 3.12, and 3.13, and on Windows with Python 3.13. It also compares
the committed sample byte for byte with a fresh run.
