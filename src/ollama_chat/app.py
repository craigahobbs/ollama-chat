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
import re
import threading
import uuid

import chisel
import ollama
import schema_markdown


# The ollama-chat back-end API WSGI application class
class OllamaChatApplication(chisel.Application):
    __slots__ = ('config', 'chats')


    def __init__(self, config_path):
        super().__init__()
        self.config = ConfigManager(config_path)
        self.chats = {}

        # Add the chisel documentation application
        self.add_requests(chisel.create_doc_requests())

        # Add the APIs
        self.add_request(delete_conversation)
        self.add_request(delete_conversation_exchange)
        self.add_request(get_conversation)
        self.add_request(get_conversations)
        self.add_request(regenerate_conversation_exchange)
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


# The ollama-chat configuration context manager
class ConfigManager:
    __slots__ = ('config_path', 'config_lock', 'config')


    DEFAULT_MODEL = 'llama3:latest'


    def __init__(self, config_path):
        self.config_path = config_path
        self.config_lock = threading.Lock()

        # Ensure the config file exists with default config if it doesn't exist
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as fh_config:
                self.config = schema_markdown.validate_type(OLLAMA_CHAT_TYPES, 'OllamaChatConfig', json.loads(fh_config.read()))
        else:
            self.config = {'model': ConfigManager.DEFAULT_MODEL, 'conversations': []}


    @contextmanager
    def __call__(self, save=False):
        # Acquire the config lock
        self.config_lock.acquire()

        try:
            # If no model is set, set the default model
            is_saving = save
            if 'model' not in self.config:
                self.config['model'] = ConfigManager.DEFAULT_MODEL
                is_saving = True

            # Yield the config on context entry
            yield self.config

            # Save the config file on context exit, if requested
            if is_saving and not self.config.get('noSave'):
                with open(self.config_path, 'w', encoding='utf-8') as fh_config:
                    json.dump(self.config, fh_config, indent=4, sort_keys = True)
        finally:
            # Release the config lock
            self.config_lock.release()


# The ollama chat manager class
class OllamaChat():
    __slots__ = ('app', 'conversation_id', 'stop')


    def __init__(self, app, conversation_id):
        self.app = app
        self.conversation_id = conversation_id
        self.stop = False

        # Start the chat thread
        chat_thread = threading.Thread(target=OllamaChat.chat_thread_fn, args=(self,))
        chat_thread.daemon = True
        chat_thread.start()


    @staticmethod
    def chat_thread_fn(chat):
        try:
            # Create the Ollama messages from the conversation
            messages = []
            with chat.app.config() as config:
                conversation = config_conversation(config, chat.conversation_id)
                model = conversation['model']
                for exchange in conversation['exchanges']:
                    messages.append({'role': 'user', 'content': exchange['user']})
                    if exchange['model'] != '':
                        messages.append({'role': 'assistant', 'content': exchange['model']})

            # Start the chat
            stream = ollama.chat(model=model, messages=messages, stream=True)

            # Stream the chat response
            for chunk in stream:
                # Stop streaming if stopped
                if chat.stop:
                    break

                # Update the conversation
                with chat.app.config() as config:
                    conversation = config_conversation(config, chat.conversation_id)
                    exchange = conversation['exchanges'][-1]
                    exchange['model'] += chunk['message']['content']

        except Exception as exc: # pylint: disable=broad-exception-caught
            # Communicate the error
            with chat.app.config() as config:
                conversation = config_conversation(config, chat.conversation_id)
                exchange = conversation['exchanges'][-1]
                exchange['model'] += f'\n**ERROR:** {exc}'

        # Save the conversation
        with chat.app.config(save=True):
            # Delete the application's chat entry
            if chat.conversation_id in chat.app.chats:
                del chat.app.chats[chat.conversation_id]


# Helper to find a conversation by ID
def config_conversation(config, id_):
    return next((conv for conv in config['conversations'] if conv['id'] == id_), None)


#
# The Ollama Chat API
#


# The Ollama Chat API type model
with pkg_resources.open_text('ollama_chat.static', 'ollamaChat.smd') as cm_smd:
    OLLAMA_CHAT_TYPES = schema_markdown.parse_schema_markdown(cm_smd.read())


@chisel.action(name='getConversations', types=OLLAMA_CHAT_TYPES)
def get_conversations(ctx, unused_req):
    models = ollama.list()['models'] or ()
    with ctx.app.config() as config:
        return {
            'model': config['model'],
            'models': sorted(model['name'] for model in models),
            'conversations': [
                {
                    'id': conversation['id'],
                    'model': conversation['model'],
                    'title': conversation['title'],
                    'generating': conversation['id'] in ctx.app.chats
                }
                for conversation in config['conversations']
            ]
        }


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
        title = re.sub(r'\s+', ' ', user_prompt).strip()
        if len(title) > max_title_len:
            title_suffix = '...'
            title = f'{title[:max_title_len - len(title_suffix)]}{title_suffix}'

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


@chisel.action(name='getConversation', types=OLLAMA_CHAT_TYPES)
def get_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Return the conversation
        return {
            'conversation': {
                **copy.deepcopy(conversation),
                'generating': id_ in ctx.app.chats
            }
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


@chisel.action(name='deleteConversationExchange', types=OLLAMA_CHAT_TYPES)
def delete_conversation_exchange(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Delete the most recent exchange (but not the last one)
        exchanges = conversation['exchanges']
        if len(exchanges) > 1:
            del exchanges[-1]


@chisel.action(name='regenerateConversationExchange', types=OLLAMA_CHAT_TYPES)
def regenerate_conversation_exchange(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Reset the most recent exchange's model response
        exchanges = conversation['exchanges']
        exchanges[-1]['model'] = ''

        # Start the model chat
        ctx.app.chats[id_] = OllamaChat(ctx.app, id_)
