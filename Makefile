# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE


# Download python-build
PYTHON_BUILD_DIR ?= ../python-build
define WGET
ifeq '$$(wildcard $(notdir $(1)))' ''
$$(info Downloading $(notdir $(1)))
_WGET := $$(shell [ -f $(PYTHON_BUILD_DIR)/$(notdir $(1)) ] && cp $(PYTHON_BUILD_DIR)/$(notdir $(1)) . || $(call WGET_CMD, $(1)))
endif
endef
WGET_CMD = if which wget; then wget -q -c $(1); else curl -f -Os $(1); fi
$(eval $(call WGET, https://craigahobbs.github.io/python-build/Makefile.base))
$(eval $(call WGET, https://craigahobbs.github.io/python-build/pylintrc))


# Set gh-pages source
GHPAGES_SRC := build/doc/


# Include python-build
include Makefile.base


# Development dependencies
TESTS_REQUIRE := bare-script


# Set the coverage limit
COVERAGE_REPORT_ARGS := $(COVERAGE_REPORT_ARGS) --fail-under 24
UNITTEST_PARALLEL_COVERAGE_ARGS := --coverage-branch --coverage-fail-under 24


# Disable pylint docstring warnings
PYLINT_ARGS := $(PYLINT_ARGS) static/models --disable=missing-class-docstring --disable=missing-function-docstring --disable=missing-module-docstring


# Don't delete models.json in gh-pages branch
GHPAGES_RSYNC_ARGS := --exclude='models/models.json'


help:
	@echo "            [run|test-app]"


clean:
	rm -rf Makefile.base pylintrc


doc:
	rm -rf $(GHPAGES_SRC)
	mkdir -p $(GHPAGES_SRC)
	cp -R \
		README.md \
		static/* \
		src/ollama_chat/static/ollamaChat.smd \
		$(GHPAGES_SRC)


.PHONY: test-app
commit: test-app
test-app: $(DEFAULT_VENV_BUILD)
	$(DEFAULT_VENV_BIN)/bare -s src/ollama_chat/static/*.bare src/ollama_chat/static/test/*.bare
	$(DEFAULT_VENV_BIN)/bare -m src/ollama_chat/static/test/runTests.bare$(if $(DEBUG), -d)$(if $(TEST), -v vTest "'$(TEST)'")


.PHONY: run
run: $(DEFAULT_VENV_BUILD)
	$(DEFAULT_VENV_BIN)/ollama-chat$(if $(ARGS), $(ARGS))


.PHONY: markdown-up
markdown-up:
	cd src/ollama_chat/static && $(call WGET_CMD, https://craigahobbs.github.io/markdown-up/markdown-up.tar.gz)
