# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat back-end application
"""

from contextlib import contextmanager
import copy
import json
import os
import importlib.resources as pkg_resources
import threading
import uuid

import chisel
import ollama
import schema_markdown

from .ollama import OllamaChat, config_conversation


# The default config
DEFAULT_CONFIG = {
    'model': 'llama3:latest',
    'conversations': []
}


class OllamaChatApplication(chisel.Application):
    """
    The ollama-chat back-end API WSGI application class
    """

    __slots__ = ('config', 'chats')


    def __init__(self, config_path):
        super().__init__()
        self.config = ConfigManager(config_path, DEFAULT_CONFIG)
        self.chats = {}

        # Add the chisel documentation application
        self.add_requests(chisel.create_doc_requests())

        # Add the APIs
        self.add_request(delete_conversation)
        self.add_request(get_conversation)
        self.add_request(get_conversations)
        self.add_request(get_model)
        self.add_request(get_models)
        self.add_request(reply_conversation)
        self.add_request(set_model)
        self.add_request(start_conversation)
        self.add_request(stop_conversation)

        # Add the ollama-chat statics
        self.add_static(
            'index.html',
            'text/html; charset=utf-8',
            (('GET', None), ('GET', '/')),
            'The Ollama Chat application HTML'
        )
        self.add_static(
            'ollamaChat.bare',
            'text/plain; charset=utf-8',
            (('GET', None),),
            'The Ollama Chat application BareScript'
        )


    def add_static(self, filename, content_type, urls, doc):
        with pkg_resources.open_binary('ollama_chat.static', filename) as fh:
            self.add_request(chisel.StaticRequest(
                filename,
                fh.read(),
                content_type=content_type,
                urls=urls,
                doc=doc,
                doc_group='Ollama Chat Statics'
            ))


class ConfigManager:
    __slots__ = ('config_path', 'config_lock', 'config')


    def __init__(self, config_path, default_config):
        self.config_path = config_path
        self.config_lock = threading.Lock()

        # Ensure the config file exists with default config if it doesn't exist
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as fh_config:
                self.config = schema_markdown.validate_type(OLLAMA_CHAT_TYPES, 'OllamaChat', json.loads(fh_config.read()))
        else:
            self.config = default_config


    @contextmanager
    def __call__(self, save=False):
        self.config_lock.acquire()
        try:
            yield self.config
            if save:
                with open(self.config_path, 'w', encoding='utf-8') as fh_config:
                    json.dump(self.config, fh_config, indent=4)
        finally:
            self.config_lock.release()


# The Ollama Chat type model
with pkg_resources.open_text('ollama_chat.static', 'ollamaChat.smd') as cm_smd:
    OLLAMA_CHAT_TYPES = schema_markdown.parse_schema_markdown(cm_smd.read())


@chisel.action(name='getModels', types=OLLAMA_CHAT_TYPES)
def get_models(unused_ctx, unused_req):
    return {
        'models': [
            {
                'model': model['name'],
                'size': model['size']
            }
            for model in ollama.list()['models']
        ]
    }


@chisel.action(name='getModel', types=OLLAMA_CHAT_TYPES)
def get_model(ctx, unused_req):
    with ctx.app.config() as config:
        return {'model': config['model']}


@chisel.action(name='setModel', types=OLLAMA_CHAT_TYPES)
def set_model(ctx, req):
    with ctx.app.config(save=True) as config:
        config['model'] = req['model']


@chisel.action(name='startConversation', types=OLLAMA_CHAT_TYPES)
def start_conversation(ctx, req):
    with ctx.app.config() as config:
        # Compute the conversation title
        user_prompt = req['user']
        max_title_len = 50
        if len(user_prompt) <= max_title_len:
            title = user_prompt
        else:
            title_suffix = '...'
            title = f'{user_prompt[:max_title_len - len(title_suffix)]}{title_suffix}'

        # Create the new conversation object
        id_ = str(uuid.uuid4())
        model = config['model']
        conversation = {
            'id': id_,
            'model': model,
            'title': title,
            'exchanges': [
                {
                    'user': req['user'],
                    'model': ''
                }
            ]
        }

        # Add the new conversation to the application config
        config['conversations'].insert(0, conversation)

        # Start the model chat
        ctx.app.chats[id_] = OllamaChat(ctx.app, id_)

        # Return the new conversation ID
        return {'id': id_}


@chisel.action(name='getConversation', types=OLLAMA_CHAT_TYPES)
def get_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Return the conversation
        return {
            'conversation': copy.deepcopy(conversation),
            'generating': id_ in ctx.app.chats
        }


@chisel.action(name='replyConversation', types=OLLAMA_CHAT_TYPES)
def reply_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Add the reply exchange
        conversation['exchanges'].append({
            'user': req['user'],
            'model': ''
        })

        # Start the model chat
        ctx.app.chats[id_] = OllamaChat(ctx.app, id_)


@chisel.action(name='stopConversation', types=OLLAMA_CHAT_TYPES)
def stop_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Not generating?
        chat = ctx.app.chats.get(id_)
        if chat is None:
            return

        # Stop the conversation
        chat.stop = True
        del ctx.app.chats[id_]


@chisel.action(name='deleteConversation', types=OLLAMA_CHAT_TYPES)
def delete_conversation(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Delete the conversation
        config['conversations'] = [conversation for conversation in config['conversations'] if conversation['id'] != id_]


@chisel.action(name='getConversations', types=OLLAMA_CHAT_TYPES)
def get_conversations(ctx, unused_req):
    conversations = []
    with ctx.app.config() as config:
        for conversation in config['conversations']:
            info = dict(conversation)
            del info['exchanges']
            conversations.append(info)
    return {
        'conversations': conversations
    }
