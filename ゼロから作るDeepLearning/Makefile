VENV := .venv

.PHONY: setup dev
# Homebrew の Python 3.14 で .venv を作り、requirements.txt のライブラリを入れる
setup:
	/opt/homebrew/bin/python3.14 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# JupyterLab を起動する
dev:
	$(VENV)/bin/jupyter lab --no-browser
