# ollama-chat

[![PyPI - Status](https://img.shields.io/pypi/status/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![PyPI](https://img.shields.io/pypi/v/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ollama-chat)](https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-chat)](https://pypi.org/project/ollama-chat/)

**Ollama Chat** is a conversational AI chat client that uses [Ollama](https://ollama.com) to interact with local large
language models (LLMs).


## Features

- Platform independent - tested on macOS, Windows, and Linux
- Chat with any local Ollama model
- Save conversations for later viewing and interaction
- Regenerate the most recent conversation response
- Delete the most recent conversation exchange
- View responses as Markdown or text
- Save conversations as Markdown text
- Multiple concurrent chats
- Prompt commands for including file and URL content
- Conversation templates for repeating prompts with variable substitutions
- Start a converstation or template from the command line
- Download and manage models


## Installation

To get up and running with Ollama Chat follows these steps:

1. Install and start [Ollama](https://ollama.com)

2. Install Ollama Chat

   ~~~
   pip install ollama-chat
   ~~~


### Updating

To update Ollama Chat:

~~~
pip install -U ollama-chat
~~~


## Start Ollama Chat

To start Ollama Chat, open a terminal prompt and run the Ollama Chat application:

~~~
ollama-chat
~~~

A web browser is launched and opens the Ollama Chat web application.

By default, a configuration file, "ollama-chat.json", is created in the user's home directory.

To start a conversation from the command line, use the `-m` argument:

~~~
ollama-chat -m "Why is the sky blue?"
~~~

To start a named template from the command line, use the `-t` and `-v` arguments:

~~~
ollama-chat -t AskAristotle -v Subject "Why is the sky blue?"
~~~


## Conversation Templates

Conversation Templates allow you to repeat the same prompts with different models. Templates can define variables for
use in the template title and prompt text (e.g., `{{var}}`).

There are two ways to create a template. Click "Add Template" from the index page, and a new template is created and
opened in the template editor. The other way is to click "Template" from a conversation view's menu.


## Prompt Commands

Ollama Chat supports special **prompt commands** that allow you to include file and URL content in
your prompt, among other things. The following prompt commands are available:

- `/file` - include a file

  ```
  /file README.md

  Please summarize the README file.
  ```

- `/dir` - include files from a directory

  ```
  /dir src/ollama_chat py

  Please provide a summary for each Ollama Chat source file.
  ```

- `/url` - include a URL resource

  ```
  /url https://craigahobbs.github.io/ollama-chat/README.md

  Please summarize the README file.
  ```

- `/do` - execute a conversation template by name or title

  ```
  /do city-report -v CityState "Seattle, WA"
  ```

To get prompt command help use the `-h` option:

```
/file -h
```


## File Format and API Documentation

[Ollama Chat File Format](https://craigahobbs.github.io/ollama-chat/api.html#var.vName='OllamaChatConfig')

[Ollama Chat API](https://craigahobbs.github.io/ollama-chat/api.html)


## Development

This package is developed using [python-build](https://github.com/craigahobbs/python-build#readme).
It was started using [python-template](https://github.com/craigahobbs/python-template#readme) as follows:

~~~
template-specialize python-template/template/ ollama-chat/ -k package ollama-chat -k name 'Craig A. Hobbs' -k email 'craigahobbs@gmail.com' -k github 'craigahobbs' -k noapi 1
~~~
