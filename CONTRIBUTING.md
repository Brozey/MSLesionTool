# Contributing

Contributions, bug reports, and platform-testing feedback are welcome.

## Dev setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/macOS: source .venv/bin/activate
python install.py --cli-only        # minimal deps without the GUI stack
python download_models.py           # fetch model weights from HuggingFace
```

## Before opening a PR

- Run `ruff check .` and `python -m compileall msseg scripts`.
- Keep changes focused; describe what you changed and how you tested it.

## Reporting issues

Open a GitHub issue with your OS, Python version, GPU/CUDA (if relevant), and steps to reproduce.
