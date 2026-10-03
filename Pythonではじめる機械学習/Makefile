VENV := .venv

.PHONY: setup dev
# Mac 標準の Python 3.9 で .venv を作り、requirements.txt のライブラリを入れる
setup:
	/usr/bin/python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# JupyterLab を起動する
dev:
	$(VENV)/bin/jupyter lab --no-browser
