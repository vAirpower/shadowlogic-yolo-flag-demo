#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  /usr/local/bin/python3.12 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

python src/export_baseline.py
python src/inject_chinese_flag.py --expose-trigger

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=src:tests pytest tests/ -v"
echo "  python src/webcam_demo.py"
