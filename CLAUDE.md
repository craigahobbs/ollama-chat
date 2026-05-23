# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Ollama Chat is a conversational AI chat client for local LLMs via [Ollama](https://ollama.com). It is
a Python WSGI back-end serving a [BareScript](https://github.com/craigahobbs/bare-script)/[MarkdownUp](https://github.com/craigahobbs/markdown-up)
client-rendered front-end. The browser loads `index.html`, which boots MarkdownUp and runs the
`.bare` application code; the front-end talks to the back-end through the JSON API.

## Commands

The build uses [python-build](https://github.com/craigahobbs/python-build) via `Makefile.base`
(downloaded automatically; not committed). All targets create/use a venv under `build/venv/`.

- `make test` — run Python unit tests (all supported Python versions when `USE_DOCKER=1`/`USE_PODMAN=1`, else system Python)
- `make test TEST=tests.test_chat.TestChat.test_name` — run a single Python test (module path is relative to `src/`)
- `make lint` — pylint over `src` (config in `pylintrc`, also downloaded)
- `make cover` — coverage; **100% coverage is enforced** (`--fail-under 100`)
- `make test-app` — run the front-end BareScript unit tests
- `make test-app TEST='testNamePrefix'` — run matching BareScript tests only
- `make run` — start the app locally (`make run ARGS="-m 'Why is the sky blue?'"` to pass CLI args)
- `make commit` — runs test + lint + doc + cover (the full pre-commit gate)
- `make clean` — remove downloaded `Makefile.base`/`pylintrc` and build artifacts

Tests require an `ollama` package dependency available as `bare-script` (see `TESTS_REQUIRE`); the
back-end never contacts a real Ollama server in tests — `urllib3` is mocked.

## Back-end architecture (`src/ollama_chat/`)

- **`app.py`** — the `OllamaChat` WSGI app (subclass of `chisel.Application`). Each API endpoint is a
  `@chisel.action` function registered in `OllamaChat.__init__`. Also registers the `.bare`/`index.html`
  statics. `ConfigManager` owns the single `ollama-chat.json` config dict behind a `threading.Lock` —
  **always access config via the `with ctx.app.config(save=...)` context manager** so writes are
  serialized and persisted (skip persistence when `noSave` is set). `DownloadManager` runs model pulls
  on a background thread.
- **`chat.py`** — `ChatManager` runs each conversation's generation on a daemon thread, streaming
  Ollama chunks into the config and re-acquiring the config lock per chunk. In-flight chats live in
  `app.chats[conversation_id]`; presence there means "generating" (used for busy checks and the
  `generating` flag). This file also implements **prompt commands** (`/file`, `/dir`, `/image`, `/url`,
  `/do`, `/?`) via a module-level `argparse` parser — `_process_commands` rewrites the user prompt and
  sets `flags` (e.g. base64 images, template expansion via `/do`).
- **`ollama.py`** — thin wrappers over the Ollama HTTP API (`/api/chat`, `/api/tags`, `/api/show`,
  `/api/pull`, `/api/delete`). Host comes from `OLLAMA_HOST` (default `http://127.0.0.1:11434`).
  `ollama_chat` detects thinking-capable models and enables `think`.
- **`main.py`** — CLI entry point (`ollama-chat`). Starts the `waitress` server and opens a browser.
  `-m`/`-t` start a conversation/template by POSTing to the API (either the in-process app or, with
  `-b`, an already-running server) and deep-link the browser to it.

## Front-end architecture (`src/ollama_chat/static/`)

- BareScript application split by view: `ollamaChat.bare` (entry/router + index page),
  `ollamaChatConversation.bare`, `ollamaChatModels.bare`, `ollamaChatTemplate.bare`, plus shared
  `ollamaChatUtil.bare`. Routing is hash-based on the `vView` arg.
- The front-end calls the back-end with `systemFetch('actionName')` and `jsonParse`. Use the
  **bare-script skill** when editing `.bare` files or front-end tests (`static/test/`).
- `ollamaChat.smd` is the [schema-markdown](https://github.com/craigahobbs/schema-markdown) type model.
  It defines both the API request/response types (validated server-side in `app.py`) and the
  `OllamaChatConfig` file format. Adding or changing an endpoint means editing the `.smd` and `app.py`
  together.

## Model metadata (`static/models/`)

`models.py` scrapes the Ollama library web page (HTML, fragile by design — no official API) to produce
`models.json`, the list of downloadable models shown in the app. The
`.github/workflows/nightly-models.yml` workflow regenerates it nightly and commits to the `gh-pages`
branch. `static/` (root) is published to GitHub Pages (`gh-pages`) via `make gh-pages`.

## Conventions

- Python classes use `__slots__`; supports Python 3.10–3.14.
- The config dict is the single source of truth for conversations/templates/current model — there is no
  database. Mutations must go through `ConfigManager`.
