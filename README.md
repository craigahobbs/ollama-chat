# ollama-chat

[![PyPI - Status](https://img.shields.io/pypi/status/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![PyPI](https://img.shields.io/pypi/v/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ollama-chat)](https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-chat)](https://pypi.org/project/ollama-chat/)

**Ollama Chat** is a conversational AI chat client that uses [Ollama](https://ollama.com) to
interact with local large language models (LLMs) entirely offline. Ideal for AI enthusiasts,
developers, or anyone wanting private, offline LLM chats.


## Features

- Chat with local large language models (LLMs) entirely offline
- ***Prompt Commands*** to include files, images, and URL content
- ***Conversation Templates*** for repeating prompts with variable substitutions
- Browse, download, monitor, and select local models directly in the app
- Multiple concurrent chats
- Regenerate the most recent conversation response
- Delete the most recent conversation exchange
- View responses as Markdown or text
- Save conversations as Markdown text
- Start a conversation or template from the command line
- Platform independent - tested on macOS, Windows, and Linux


## Installation

To get up and running with Ollama Chat follow these steps:

1. Install and start [Ollama](https://ollama.com)

2. Install Ollama Chat

   ~~~
   pip install ollama-chat
   ~~~


## Start Ollama Chat

To start Ollama Chat, open a terminal prompt and run the Ollama Chat application:

~~~
ollama-chat
~~~

A web browser is launched and opens the Ollama Chat application.

By default, a configuration file, "ollama-chat.json", is created in the user's home directory.


### Add a Desktop Launcher

To add a desktop launcher, follow the steps for your OS.


#### macOS

In Finder, locate the ollama-chat executable and drag-and-drop it into the lower portion of the Dock.


#### Windows

In File Explorer, locate the ollama-chat executable, right-click it, and select "Pin to Start".


#### GNOME (Linux)

Execute the following command in a shell:

~~~
wget https://craigahobbs.github.io/ollama-chat/ollama-chat.desktop -P $HOME/.local/share/applications
~~~


### Start a Conversation from the Command Line

To start a conversation from the command line, use the `-m` argument:

~~~
ollama-chat -m "Why is the sky blue?"
~~~


### Start a Template from the Command Line

To start a named template from the command line, use the `-t` and `-v` arguments:

~~~
ollama-chat -t askAristotle -v question "Why is the sky blue?"
~~~


## Conversation Templates

Conversation Templates allow you to repeat the same prompts with different models. Templates can define variables for
use in the template title and prompt text (e.g., `{{var}}`).

There are two ways to create a template. Click "Add Template" from the index page, and a new template is created and
opened in the template editor. The other way is to click "Template" from a conversation view's menu.


## Prompt Commands

Ollama Chat supports special **prompt commands** that allow you to include files, images, and URL content in
your prompt, among other things. The following prompt commands are available:

- `/file` - include a file

  ```
  /file README.md

  Please summarize the README file.
  ```

- `/image` - include an image

  ```
  /image image.jpeg

  Please summarize the image.
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

- `/do` - execute a conversation template by name

  ```
  /do city-report -v CityState "Seattle, WA"
  ```

- `/?` - list available prompt commands

  ```
  /?
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
