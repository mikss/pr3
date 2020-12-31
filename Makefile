# bash shell setup
SHELL=/bin/bash -o errexit -o nounset -o pipefail
export GIT_SHA=$(shell git rev-parse HEAD)

.PHONY: bootstrap
bootstrap: .git/.initialized_brew
bootstrap: .git/.initialized_direnv
bootstrap:
	@DIRENV_LOG_FORMAT="" /usr/local/bin/direnv exec . $(MAKE) _bootstrap --no-print-directory
	@echo "Bootstrapping done!"

.PHONY: _bootstrap
_bootstrap: .venv/
_bootstrap: pip-sync
_bootstrap: .git/.initialized_pre_commit

.git/.initialized_brew: Brewfile
	if ! hash brew; then \
		/usr/bin/ruby -e "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"; \
	fi
	/usr/local/bin/brew update
	/usr/local/bin/brew bundle --no-lock
	@# Update the virtualenv symlinks in case the python version updated.
	@#
	@# Using "python3 -m venv" doesn't recreate the .venv/lib symlinks, but "virtualenv" does...
	if [[ -d .venv ]]; then \
		echo "Deleting .venv symlinks..." && \
		find .venv -type l -delete && \
		echo "Recreating .venv symlinks..." && \
		/usr/local/bin/pip3 install virtualenv && \
		/usr/local/bin/virtualenv .venv; \
	fi;
	pip3 install --upgrade pip
	touch $@

.git/.initialized_direnv: .envrc
	/usr/local/bin/direnv allow
	touch $@

.git/.initialized_pre_commit: .pre-commit-config.yaml
	pre-commit install
	pre-commit install -t pre-push
	pre-commit install-hooks
	touch $@

.venv/:
	/usr/local/bin/python3 -m venv .venv/


.PHONY: pip-compile
pip-compile: */*.in
	@if ! grep "$$(sha256sum */*.in | sha256sum)" requirements.txt > /dev/null; then \
		pip-compile -q --output-file requirements.txt */*.in; \
		echo -e "\n# input sha: $$(sha256sum */*.in | sha256sum)" >> requirements.txt; \
	fi

.PHONY: pip-sync
pip-sync: pip-compile
	@if [[ -f requirements/dev.in.override ]]; then \
		FILES="requirements.txt requirements/dev.in.override"; \
	else \
		FILES="requirements.txt"; \
	fi; \
	if ! grep "$$(sha256sum $$FILES | sha256sum)" .git/.last_pip_sync > /dev/null; then \
		pip3 install $$(grep '^pip' requirements/dev.in); \
		pip-sync $$FILES; \
	fi; \
	echo "$$FILES sha: $$(sha256sum $$FILES | sha256sum)" > .git/.last_pip_sync; \


.PHONY: prep-commit
prep-commit:
	git diff --name-only HEAD | xargs pre-commit run --files
