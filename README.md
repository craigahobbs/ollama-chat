# ollama-chat

[![PyPI - Status](https://img.shields.io/pypi/status/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![PyPI](https://img.shields.io/pypi/v/ollama-chat)](https://pypi.org/project/ollama-chat/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ollama-chat)](https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-chat)](https://pypi.org/project/ollama-chat/)

Coming soon!


## Features

- Index page

  - Current model: <model> [Select]

    - Selection page lists downloaded models with select and delete links

    - Model selection page allows downloading models and reports status of downloads

    - Model download page lists models available to download

      - https://huggingface.co/api/models

   - Start new chat link

   - List of conversation links with delete links

- Conversation page

  - Chat title with edit title link

  - Chat model: <model>

  - Chat messages in scrollable region

    - Latest message has delete and regenerate buttons

  - Message input and send/stop button rendered at top and positioned below messages

  - Link to download conversation as Markdown file

- Details

  - Selected model and conversations are save to the ollama-chat.json file in the current directory


## Future Features

- Message review prompt and sanity check response buttons

- Prompt library


## Development

This package is developed using [python-build](https://github.com/craigahobbs/python-build#readme).
It was started using [python-template](https://github.com/craigahobbs/python-template#readme) as follows:

~~~
template-specialize python-template/template/ ollama-chat/ -k package ollama-chat -k name 'Craig A. Hobbs' -k email 'craigahobbs@gmail.com' -k github 'craigahobbs' -k noapi 1
~~~
