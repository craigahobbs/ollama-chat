# ollama-chat

[![PyPI - Status](https://img.shields.io/pypi/status/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![PyPI](https://img.shields.io/pypi/v/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ollama-chat)](https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-chat)](https://pypi.org/project/ollama-chat/)

**Ollama Chat** is a simple yet useful web chat client for
[Ollama](https://ollama.com)
that allows you to chat locally (and privately) with
[open-source LLMs](https://ollama.com/library).


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


## Start Conversation from CLI

To start a conversation from the command line, use the `-m` argument:

~~~
ollama-chat -m "Why is the sky blue?"
~~~


## File Format and API Documentation

[Ollama Chat File Format](https://craigahobbs.github.io/ollama-chat/api.html#var.vName='OllamaChatConfig')

[Ollama Chat API](https://craigahobbs.github.io/ollama-chat/api.html)


## Future

- Auto-title task on start conversation
  - Update conversation title API
  - Update title link on index/conversation page

- Prompts part 1
  - Prompts config collection (name, title, prompt)
  - Index links start new conversation with current model
  - `-t` command-line argument starts prompt by name

- File / Directory / URL text inclusion in prompt

- Prompts part 2
  - Prompt editor
  - Create link on index page
  - Delete links on index page
  - Index links open template editor if any template markers (e.g. "{Name}")

- Local model management (pull, rm)
  - [Models JSON](https://huggingface.co/api/models)


## Development

This package is developed using [python-build](https://github.com/craigahobbs/python-build#readme).
It was started using [python-template](https://github.com/craigahobbs/python-template#readme) as follows:

~~~
template-specialize python-template/template/ ollama-chat/ -k package ollama-chat -k name 'Craig A. Hobbs' -k email 'craigahobbs@gmail.com' -k github 'craigahobbs' -k noapi 1
~~~
