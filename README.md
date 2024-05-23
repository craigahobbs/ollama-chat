# ollama-chat

[![PyPI - Status](https://img.shields.io/pypi/status/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![PyPI](https://img.shields.io/pypi/v/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ollama-chat)](https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-chat)](https://pypi.org/project/ollama-chat/)

**Ollama Chat** is a simple yet useful web chat client for
[Ollama](https://ollama.com)
that allows you to chat locally (and privately) with
[open-source LLMs](https://ollama.com/library).


# Installation

To get up and running with Ollama Chat follows these steps:

1. Install and start [Ollama](https://ollama.com)

2. Install Ollama Chat

   ~~~
   pip install ollama-chat
   ~~~


# Starting Ollama Chat

To start Ollama Chat, open a terminal prompt and run the Ollama Chat application:

~~~
ollama-chat
~~~

A web browser is launched and opens the Ollama Chat web application.

By default, a configuration file, "ollama-chat.json", is created in the current directory to save
your conversations.


## Future Features

In no particular order...

- Save conversation as Markdown file

- Conversation title edit

- File / Directory / URL text inclusion in prompt

- Local model management (pull, rm)
  - [Models JSON](https://huggingface.co/api/models)

- Prompt library


## Development

This package is developed using [python-build](https://github.com/craigahobbs/python-build#readme).
It was started using [python-template](https://github.com/craigahobbs/python-template#readme) as follows:

~~~
template-specialize python-template/template/ ollama-chat/ -k package ollama-chat -k name 'Craig A. Hobbs' -k email 'craigahobbs@gmail.com' -k github 'craigahobbs' -k noapi 1
~~~
