# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

# Download python-build
define WGET
ifeq '$$(wildcard $(notdir $(1)))' ''
$$(info Downloading $(notdir $(1)))
_WGET := $$(shell $(call WGET_CMD, $(1)))
endif
endef
WGET_CMD = if which wget; then wget -q -c $(1); else curl -f -Os $(1); fi
$(eval $(call WGET, https://raw.githubusercontent.com/craigahobbs/python-build/main/Makefile.base))
$(eval $(call WGET, https://raw.githubusercontent.com/craigahobbs/python-build/main/pylintrc))


# Set gh-pages source
GHPAGES_SRC := build/doc/


# Development dependencies
TESTS_REQUIRE := bare-script


# Include python-build
include Makefile.base


# Set the coverage limit
COVERAGE_REPORT_ARGS := $(COVERAGE_REPORT_ARGS) --fail-under 25


# Disable pylint docstring warnings
PYLINT_ARGS := $(PYLINT_ARGS) --disable=missing-class-docstring --disable=missing-function-docstring --disable=missing-module-docstring


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
	$(DEFAULT_VENV_BIN)/bare -s src/ollama_chat/static/*.bare


.PHONY: run
run: $(DEFAULT_VENV_BUILD)
	$(DEFAULT_VENV_BIN)/ollama-chat
