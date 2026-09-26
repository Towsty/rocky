# Rocky brain (desk-robot vendor)

Upstream: https://github.com/cgro00/desk-robot (MIT) — see `LICENSE.desk-robot` at repo root.

## Run

```bash
cd server
# Prefer Python 3.13 (e.g. `uv python install 3.13`); 3.12+ also runs.
python3.13 -m venv .venv   # or: python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill LLM_API_KEY (and optional FISH_AUDIO_API_KEY, HUMAN_NAME, ROBOT_TOKEN)
python -m brain.main
```

Then type `ask what is a weekend?` (typed ask is the debug hatch). Live face/controls: http://localhost:8766/

No hardware required. Mic wake ("hey Rocky") needs a real mic — path is ready; cloud VMs usually block that acceptance.

Without `LLM_API_KEY`, ask answers `Brain has no key` (no stack trace). Idle uses no LLM and no Fish.
