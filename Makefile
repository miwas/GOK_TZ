.PHONY: clean help install env serve # lint test
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

env: ## create Python virtual environment
	test -d .venv || python3 -m venv .venv

install: env ## install packages to .venv
	. .venv/bin/activate && pip install --upgrade pip &&\
			 pip install -r requirements.txt &&\
					pip install -r requirements-dev.txt

clean: ## remove Python file artifacts
	rm -rf .venv

serve: ## launch Bokeh server
	bokeh serve --dev --show viz/ # med_capital.py med_parts.py #  
