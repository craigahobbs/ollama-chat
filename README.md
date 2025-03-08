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
- Show/hide thinking of reasoning models
- Regenerate the most recent conversation response
- Delete the most recent conversation exchange
- View responses as Markdown or text
- Save conversations as Markdown text
- Start a conversation or template from the command line
- Platform independent - tested on macOS, Windows, and Linux


## Installation

To get up and running with Ollama Chat follow these steps:

1. Install [Ollama](https://ollama.com/download)

2. Install Ollama Chat

   **macOS and Linux**

   ~~~
   python3 -m venv $HOME/venv --upgrade-deps
   . $HOME/venv/bin/activate
   pip install ollama-chat
   ~~~

   **Windows**

   ~~~
   python3 -m venv %USERPROFILE%\venv --upgrade-deps
   %USERPROFILE%\venv\Scripts\activate
   pip install ollama-chat
   ~~~


## Start Ollama Chat

To start Ollama Chat, open a terminal prompt and follow the steps for your OS. When you start Ollama
Chat, a web browser is launched and opens the Ollama Chat application.

By default, a configuration file, "ollama-chat.json", is created in the user's home directory.

### macOS and Linux

~~~
. $HOME/venv/bin/activate
ollama-chat
~~~

### Windows

~~~
%USERPROFILE%\venv\Scripts\activate
ollama-chat
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


## Add a Desktop Launcher

To add a desktop launcher, follow the steps for your OS.


### macOS

In Finder, locate the "ollama-chat" executable and drag-and-drop it into the lower portion of the
Dock.


### Windows

In File Explorer, locate the "ollama-chat" executable, right-click it, and select "Pin to Start".


### GNOME (Linux)

1. Copy the following Ollama Chat desktop file contents:

   ~~~
   [Desktop Entry]
   Name=Ollama Chat
   Exec=sh -c "$HOME/venv/bin/ollama-chat"
   Type=Application
   Icon=dialog-information
   Terminal=true
   Categories=Utility;
   ~~~

2. Create the Ollama Chat desktop file and paste the contents:

   ~~~
   nano $HOME/.local/share/applications/ollama-chat.desktop
   ~~~

3. Update the "Exec" path, if necessary, and save


## Conversation Templates

Conversation Templates allow you to repeat a sequence of prompts. Templates can include variable
substitutions in the title text and the prompt text (e.g., `{{var}}`).


### Create a Template

There are two ways to create a template:

- Click "Add Template" from the home page
- Click "Template" on a conversation page


### Run a Template

To run a template, click on its title on the home page. If the template has any variables, the user
is prompted for their values prior to running the template. When a template runs, a new conversation
is created and each prompt is entered in sequence.


### Edit a Template

To edit a template, from the home page, click "Select" on the template you want to edit, and then
click "Edit". On the template editor page you can update the template's title, set its name, add or
remove variables, and add or remove prompts.


## Prompt Commands

Ollama Chat supports special **prompt commands** that allow you to include files, images, and URL
content in your prompt, among other things. The following prompt commands are available:

- `/file` - include a file

  ```
  /file README.md
  ```

- `/image` - include an image

  ```
  /image image.jpeg
  ```

- `/dir` - include files from a directory

  ```
  /dir src/ollama_chat py
  ```

- `/url` - include a URL resource

  ```
  /url https://craigahobbs.github.io/ollama-chat/README.md
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
