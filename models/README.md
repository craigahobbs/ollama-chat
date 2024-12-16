# The Models JSON Script

This directory contains the script that generates the `models.json` file. The models JSON file
contains the list of models available to download via Ollama. It is intended to be run periodically
to update the available models and their statistics.

Ideally, the information contained in the models JSON file would be provided by an Ollama publically
available API (static resource). Unfortunately, no such API exists at the time of this writing.

[Models JSON File Format](https://craigahobbs.github.io/ollama-chat/api.html#var.vName='OllamaChatModels')


## Implementation

The script, `models.py`, downloads the Ollama models web page and scrapes it for model names,
parameter counts, downloads, and last modified dates. If this sounds fragile to you, you are right.
