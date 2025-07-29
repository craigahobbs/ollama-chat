# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import datetime
import json
import os

import urllib3


def _get_ollama_host():
    return os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')


def ollama_chat(pool_manager, model, messages):
    # Is this a thinking model?
    url_show = f'{_get_ollama_host()}/api/show'
    data_show = {'model': model}
    response_show = pool_manager.request('POST', url_show, json=data_show)
    if response_show.status >= 400:
        raise urllib3.exceptions.HTTPError(f'Unknown model "{model}"')
    model_show = response_show.json()
    is_thinking = 'capabilities' in model_show and 'thinking' in model_show['capabilities']

    # Start a streaming chat request
    url_chat = f'{_get_ollama_host()}/api/chat'
    data_chat = {'model': model, 'messages': messages, 'stream': True, 'think': is_thinking}
    response_chat = pool_manager.request('POST', url_chat, json=data_chat, preload_content=False)
    if response_chat.status >= 400:
        raise urllib3.exceptions.HTTPError(f'Unknown model "{model}"')

    # Respond with each streamed JSON chunk
    yield from (json.loads(line.decode('utf-8')) for line in response_chat.read_chunked())


def ollama_list(pool_manager):
    url_list = f'{_get_ollama_host()}/api/tags'
    response_list = pool_manager.request('GET', url_list)
    if response_list.status >= 400:
        raise urllib3.exceptions.HTTPError(f'HTTP Error {response_list.status}')
    return [
        {
            'model': model['model'],
            'details': model['details'],
            'size': model['size'],
            'modified_at': datetime.datetime.fromisoformat(model['modified_at'])
        }
        for model in response_list.json()['models']
    ]


def ollama_delete(pool_manager, model):
    url_delete = f'{_get_ollama_host()}/api/delete'
    data_delete = {'model': model}
    response_delete = pool_manager.request('DELETE', url_delete, json=data_delete)
    if response_delete.status >= 400:
        raise urllib3.exceptions.HTTPError(f'Unknown model {model}')


def ollama_pull(pool_manager, model):
    url_pull = f'{_get_ollama_host()}/api/pull'
    data_pull = {'model': model, 'stream': True}
    response_pull = pool_manager.request('POST', url_pull, json=data_pull, preload_content=False)
    if response_pull.status >= 400:
        raise urllib3.exceptions.HTTPError(f'Unknown model {model}')

    # Respond with each streamed JSON chunk
    yield from (json.loads(line.decode('utf-8')) for line in response_pull.read_chunked())
