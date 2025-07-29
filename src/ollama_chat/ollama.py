# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import datetime
import json
import os


def _get_ollama_host():
    return os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')


def ollama_chat(session, model, messages):
    # Is this a thinking model?
    url_show = f'{_get_ollama_host()}/api/show'
    data_show = {'model': model}
    response_show = session.post(url_show, json=data_show)
    response_show.raise_for_status()
    model_show = response_show.json()
    is_thinking = 'capabilities' in model_show and 'thinking' in model_show['capabilities']

    # Start a streaming chat request
    url_chat = f'{_get_ollama_host()}/api/chat'
    data_chat = {'model': model, 'messages': messages, 'stream': True, 'think': is_thinking}
    response_chat = session.post(url_chat, json=data_chat, stream=True)
    response_chat.raise_for_status()

    # Respond with each streamed JSON chunk
    yield from (json.loads(line.decode('utf-8')) for line in response_chat.iter_lines())


def ollama_list(session):
    url_list = f'{_get_ollama_host()}/api/tags'
    response_list = session.get(url_list)
    response_list.raise_for_status()
    return [
        {
            'model': model['model'],
            'details': model['details'],
            'size': model['size'],
            'modified_at': datetime.datetime.fromisoformat(model['modified_at'])
        }
        for model in response_list.json()['models']
    ]


def ollama_delete(session, model):
    url_delete = f'{_get_ollama_host()}/api/delete'
    data_delete = {'model': model}
    response_delete = session.delete(url_delete, json=data_delete)
    response_delete.raise_for_status()


def ollama_pull(session, model):
    url_pull = f'{_get_ollama_host()}/api/pull'
    data_pull = {'model': model, 'stream': True}
    response_pull = session.post(url_pull, json=data_pull, stream=True)
    response_pull.raise_for_status()

    # Respond with each streamed JSON chunk
    yield from (json.loads(line.decode('utf-8')) for line in response_pull.iter_lines())
