## Contributing

Thank you for your interest in contributing! This repo demonstrates production-quality practices; please help keep it tidy and reproducible.

- Use Conventional Commits (e.g., `feat: add AE mapping`)
- Run `pre-commit install` once; hooks will run on every commit
- Ensure `make lint typecheck security test` passes before opening PRs
- Keep functions small, typed, and documented; avoid hard-coded paths
- Add/adjust tests to keep coverage ≥ 85%

### Dev quickstart

1. `make setup`
2. `make data`
3. `make demo`

### Code style

- Formatter: `black`
- Lint: `ruff`
- Types: `mypy`
- Security: `bandit`

