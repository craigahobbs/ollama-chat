# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat back-end application
"""

from contextlib import contextmanager
import copy
import json
import os
import importlib.resources
import re
import threading
import uuid

import chisel
import ollama
import schema_markdown

from .chat import ChatManager, config_conversation, config_template_prompts


# The ollama-chat back-end API WSGI application class
class OllamaChat(chisel.Application):
    __slots__ = ('config', 'chats')


    def __init__(self, config_path):
        super().__init__()
        self.config = ConfigManager(config_path)
        self.chats = {}

        # Add the chisel documentation application
        self.add_requests(chisel.create_doc_requests())

        # Add the APIs
        self.add_request(create_template_from_conversation)
        self.add_request(delete_conversation)
        self.add_request(delete_conversation_exchange)
        self.add_request(delete_template)
        self.add_request(get_conversation)
        self.add_request(get_conversations)
        self.add_request(get_template)
        self.add_request(move_conversation)
        self.add_request(move_template)
        self.add_request(regenerate_conversation_exchange)
        self.add_request(reply_conversation)
        self.add_request(set_conversation_title)
        self.add_request(set_model)
        self.add_request(start_conversation)
        self.add_request(start_template)
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
            'The Ollama Chat application'
        )
        self.add_static(
            'ollamaChatConversation.bare',
            'text/plain; charset=utf-8',
            (('GET', None),),
            'The Ollama Chat application conversation page'
        )


    def add_static(self, filename, content_type, urls, doc, doc_group='Ollama Chat Statics'):
        with importlib.resources.files('ollama_chat.static').joinpath(filename).open('rb') as fh:
            self.add_request(chisel.StaticRequest(filename, fh.read(), content_type, urls, doc, doc_group))


# The ollama-chat configuration context manager
class ConfigManager:
    __slots__ = ('config_path', 'config_lock', 'config')


    DEFAULT_MODEL = 'llama3.1:latest'


    def __init__(self, config_path):
        self.config_path = config_path
        self.config_lock = threading.Lock()

        # Ensure the config file exists with default config if it doesn't exist
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as fh_config:
                self.config = schema_markdown.validate_type(OLLAMA_CHAT_TYPES, 'OllamaChatConfig', json.loads(fh_config.read()))
        else:
            self.config = {'conversations': []}


    @contextmanager
    def __call__(self, save=False):
        # Acquire the config lock
        self.config_lock.acquire()

        try:
            # Yield the config on context entry
            yield self.config

            # Save the config file on context exit, if requested
            if save and not self.config.get('noSave'):
                with open(self.config_path, 'w', encoding='utf-8') as fh_config:
                    json.dump(self.config, fh_config, indent=4, sort_keys = True)
        finally:
            # Release the config lock
            self.config_lock.release()


    @classmethod
    def get_model(cls, config):
        return config.get('model', cls.DEFAULT_MODEL)


# The Ollama Chat API type model
with importlib.resources.files('ollama_chat.static').joinpath('ollamaChat.smd').open('r') as cm_smd:
    OLLAMA_CHAT_TYPES = schema_markdown.parse_schema_markdown(cm_smd.read())


@chisel.action(name='getConversations', types=OLLAMA_CHAT_TYPES)
def get_conversations(ctx, unused_req):
    try:
        models = ollama.list()['models'] or ()
    except: # pylint: disable=bare-except
        models = ()
    with ctx.app.config() as config:
        return {
            'model': ConfigManager.get_model(config),
            'models': sorted(model['name'] for model in models),
            'conversations': [
                {
                    'id': conversation['id'],
                    'model': conversation['model'],
                    'title': conversation['title']
                }
                for conversation in config['conversations']
            ],
            'templates': [
                {
                    'id': template['id'],
                    'title': template['title']
                }
                for template in (config.get('templates') or ())
            ]
        }


@chisel.action(name='setModel', types=OLLAMA_CHAT_TYPES)
def set_model(ctx, req):
    with ctx.app.config(save=True) as config:
        config['model'] = req['model']


@chisel.action(name='moveConversation', types=OLLAMA_CHAT_TYPES)
def move_conversation(ctx, req):
    with ctx.app.config(save=True) as config:
        # Find the conversation index
        id_ = req['id']
        conversations = config['conversations']
        ix_conv = next((ix for ix, conv in enumerate(conversations) if conv['id'] == id_), None)
        if ix_conv is None:
            raise chisel.ActionError('UnknownConversationID')
        conversation = conversations[ix_conv]

        # Move down?
        if req['down']:
            if ix_conv < len(conversations) - 1:
                conversations[ix_conv] = conversations[ix_conv + 1]
                conversations[ix_conv + 1] = conversation
        else:
            if ix_conv > 0:
                conversations[ix_conv] = conversations[ix_conv - 1]
                conversations[ix_conv - 1] = conversation


@chisel.action(name='moveTemplate', types=OLLAMA_CHAT_TYPES)
def move_template(ctx, req):
    with ctx.app.config(save=True) as config:
        # Find the template index
        id_ = req['id']
        templates = config['templates'] or []
        ix_tmpl = next((ix for ix, tmpl in enumerate(templates) if tmpl['id'] == id_), None)
        if ix_tmpl is None:
            raise chisel.ActionError('UnknownTemplateID')
        template = templates[ix_tmpl]

        # Move down?
        if req['down']:
            if ix_tmpl < len(templates) - 1:
                templates[ix_tmpl] = templates[ix_tmpl + 1]
                templates[ix_tmpl + 1] = template
        else:
            if ix_tmpl > 0:
                templates[ix_tmpl] = templates[ix_tmpl - 1]
                templates[ix_tmpl - 1] = template


@chisel.action(name='deleteTemplate', types=OLLAMA_CHAT_TYPES)
def delete_template(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        templates = config['templates'] or []
        ix_tmpl = next((ix for ix, tmpl in enumerate(templates) if tmpl['id'] == id_), None)
        if ix_tmpl is None:
            raise chisel.ActionError('UnknownTemplateID')
        del templates[ix_tmpl]


@chisel.action(name='getTemplate', types=OLLAMA_CHAT_TYPES)
def get_template(ctx, req):
    template_id = req['id']
    with ctx.app.config(save=True) as config:
        templates = config.get('templates') or []
        template = next((template for template in templates if template['id'] == template_id), None)
        if template is None:
            raise chisel.ActionError('UnknownTemplateID')
        return copy.deepcopy(template)


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
        conversation = {
            'id': id_,
            'model': ConfigManager.get_model(config),
            'title': title,
            'exchanges': []
        }

        # Add the new conversation to the application config
        config['conversations'].insert(0, conversation)

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, [user_prompt])

        # Return the new conversation identifier
        return {'id': id_}


@chisel.action(name='startTemplate', types=OLLAMA_CHAT_TYPES)
def start_template(ctx, req):
    template_id = req['id']
    variable_values = req.get('variables') or {}

    with ctx.app.config() as config:
        # Get the conversation template
        templates = config.get('templates') or []
        template = next((template for template in templates if template['id'] == template_id), None)
        if template is None:
            raise chisel.ActionError('UnknownTemplateID')

        # Get the template prompts
        try:
            title, prompts = config_template_prompts(template, variable_values)
        except ValueError as exc:
            message = str(exc)
            error = 'UnknownVariable' if message.startswith('unknown') else 'MissingVariable'
            raise chisel.ActionError(error, message)

        # Create the new conversation object
        id_ = str(uuid.uuid4())
        conversation = {
            'id': id_,
            'model': ConfigManager.get_model(config),
            'title': title,
            'exchanges': []
        }

        # Add the new conversation to the application config
        config['conversations'].insert(0, conversation)

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, prompts)

        # Return the new conversation identifier
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

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, [req['user']])


@chisel.action(name='setConversationTitle', types=OLLAMA_CHAT_TYPES)
def set_conversation_title(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Set the conversation title
        conversation['title'] = req['title']


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


@chisel.action(name='createTemplateFromConversation', types=OLLAMA_CHAT_TYPES)
def create_template_from_conversation(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Create the new template
        id_ = str(uuid.uuid4())
        template = {
            'id': id_,
            'title': conversation['title'],
            'prompts': [exchange['user'] for exchange in conversation['exchanges']]
        }

        # A conversation's exchanges can be empty, a template's prompts cannot
        if not template['prompts']:
            template['prompts'].append('')

        # Add the new template to the application config
        if 'templates' not in config:
            config['templates'] = []
        config['templates'].insert(0, template)

        # Return the new template identifier
        return {'id': id_}


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

        # Delete the most recent exchange
        exchanges = conversation['exchanges']
        if len(exchanges):
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

        # Any exchanges?
        exchanges = conversation['exchanges']
        if len(exchanges):
            # Delete the most recent exchange
            prompt = exchanges[-1]['user']
            del exchanges[-1]

            # Start the model chat
            ctx.app.chats[id_] = ChatManager(ctx.app, id_, [prompt])
