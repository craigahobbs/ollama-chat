# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import datetime
import json
import os

import urllib3


def _get_ollama_url(path):
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return f'{ollama_host}{path}'


def ollama_chat(pool_manager, model, messages):
    # Is this a thinking model?
    url_show = _get_ollama_url('/api/show')
    data_show = {'model': model}
    response_show = pool_manager.request('POST', url_show, json=data_show, retries=0)
    try:
        if response_show.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_show.status})')
        model_show = response_show.json()
    finally:
        response_show.close()
    is_thinking = 'capabilities' in model_show and 'thinking' in model_show['capabilities']

    # Start a streaming chat request
    url_chat = _get_ollama_url('/api/chat')
    data_chat = {'model': model, 'messages': messages, 'stream': True, 'think': is_thinking}
    response_chat = pool_manager.request('POST', url_chat, json=data_chat, preload_content=False, retries=0)
    try:
        if response_chat.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_chat.status})')

        # Respond with each streamed JSON chunk
        for chunk in (json.loads(line.decode('utf-8')) for line in response_chat.read_chunked()):
            if 'error' in chunk:
                raise urllib3.exceptions.HTTPError(chunk['error'])
            yield chunk
    finally:
        response_chat.close()


def ollama_list(pool_manager):
    url_list = _get_ollama_url('/api/tags')
    response_list = pool_manager.request('GET', url_list, retries=0)
    try:
        if response_list.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unexpected error ({response_list.status})')
        return [
            {
                'model': model['model'],
                'details': model['details'],
                'size': model['size'],
                'modified_at': datetime.datetime.fromisoformat(model['modified_at'])
            }
            for model in response_list.json()['models']
        ]
    finally:
        response_list.close()


def ollama_delete(pool_manager, model):
    url_delete = _get_ollama_url('/api/delete')
    data_delete = {'model': model}
    response_delete = pool_manager.request('DELETE', url_delete, json=data_delete, retries=0)
    try:
        if response_delete.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_delete.status})')
    finally:
        response_delete.close()


def ollama_pull(pool_manager, model):
    url_pull = _get_ollama_url('/api/pull')
    data_pull = {'model': model, 'stream': True}
    response_pull = pool_manager.request('POST', url_pull, json=data_pull, preload_content=False, retries=0)
    try:
        if response_pull.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_pull.status})')

        # Respond with each streamed JSON chunk
        yield from (json.loads(line.decode('utf-8')) for line in response_pull.read_chunked())
    finally:
        response_pull.close()
