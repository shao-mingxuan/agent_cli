.PHONY: chat run install

chat:
	venv/bin/agent chat

run:
	venv/bin/agent run -p "$(p)"

install:
	venv/bin/python -m pip install -e .
