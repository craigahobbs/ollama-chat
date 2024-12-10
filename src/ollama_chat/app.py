# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat back-end application
"""

from contextlib import contextmanager
import copy
import ctypes
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
    __slots__ = ('config', 'chats', 'downloads')


    def __init__(self, config_path):
        super().__init__()
        self.config = ConfigManager(config_path)
        self.chats = {}
        self.downloads = {}

        # Add the chisel documentation application
        self.add_requests(chisel.create_doc_requests())

        # Add the APIs
        self.add_request(create_template)
        self.add_request(delete_conversation)
        self.add_request(delete_conversation_exchange)
        self.add_request(delete_model)
        self.add_request(delete_template)
        self.add_request(download_model)
        self.add_request(get_conversation)
        self.add_request(get_conversations)
        self.add_request(get_models)
        self.add_request(get_system_info)
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
        self.add_request(stop_model_download)
        self.add_request(update_template)

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
        self.add_static(
            'ollamaChatModels.bare',
            'text/plain; charset=utf-8',
            (('GET', None),),
            'The Ollama Chat application models page'
        )
        self.add_static(
            'ollamaChatTemplate.bare',
            'text/plain; charset=utf-8',
            (('GET', None),),
            'The Ollama Chat application template page'
        )


    def add_static(self, filename, content_type, urls, doc, doc_group='Ollama Chat Statics'):
        with importlib.resources.files('ollama_chat.static').joinpath(filename).open('rb') as fh:
            self.add_request(chisel.StaticRequest(filename, fh.read(), content_type, urls, doc, doc_group))


# The ollama-chat configuration context manager
class ConfigManager:
    __slots__ = ('config_path', 'config_lock', 'config')


    DEFAULT_MODEL = 'llama3.2:latest'


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


# The model download manager class
class DownloadManager():
    __slots__ = ('app', 'model', 'status', 'completed', 'total', 'stop')


    def __init__(self, app, model):
        self.app = app
        self.model = model
        self.status = ''
        self.completed = 0
        self.total = 0
        self.stop = False

        # Start the download thread
        download_thread = threading.Thread(target=self.download_thread_fn, args=(self,))
        download_thread.daemon = True
        download_thread.start()


    @staticmethod
    def download_thread_fn(manager):
        try:
            for progress in ollama.pull(manager.model, stream=True):
                # Stopped?
                if manager.stop:
                    break

                # Update the download status
                manager.status = progress.status
                manager.completed = progress.completed or 0
                manager.total = progress.total

        except Exception: # pylint: disable=broad-exception-caught
            pass

        # Delete the application's download entry
        if manager.model in manager.app.downloads:
            del manager.app.downloads[manager.model]


# The Ollama Chat API type model
with importlib.resources.files('ollama_chat.static').joinpath('ollamaChat.smd').open('r') as cm_smd:
    OLLAMA_CHAT_TYPES = schema_markdown.parse_schema_markdown(cm_smd.read())


@chisel.action(name='getConversations', types=OLLAMA_CHAT_TYPES)
def get_conversations(ctx, unused_req):
    with ctx.app.config() as config:
        return {
            'model': ConfigManager.get_model(config),
            'conversations': [
                {
                    'id': conversation['id'],
                    'model': conversation['model'],
                    'title': conversation['title']
                }
                for conversation in config['conversations']
            ],
            'templates': [
                _create_template_info(template)
                for template in (config.get('templates') or ())
            ]
        }


def _create_template_info(template):
    template_info = {
        'id': template['id'],
        'title': template['title']
    }
    if 'name' in template:
        template_info['name'] = template['name']
    return template_info


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


@chisel.action(name='updateTemplate', types=OLLAMA_CHAT_TYPES)
def update_template(ctx, req):
    template_id = req['id']
    with ctx.app.config(save=True) as config:
        templates = config.get('templates') or []
        ix_template = next((ix_tmpl for ix_tmpl, tmpl in enumerate(templates) if tmpl['id'] == template_id), None)
        if ix_template is None:
            raise chisel.ActionError('UnknownTemplateID')
        templates[ix_template] = req


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


@chisel.action(name='createTemplate', types=OLLAMA_CHAT_TYPES)
def create_template(ctx, req):
    with ctx.app.config(save=True) as config:
        # Create the new template
        id_ = str(uuid.uuid4())
        template = {
            'id': id_,
            'title': req['title'],
            'prompts': req['prompts']
        }
        if 'name' in req:
            template['name'] = req['name']
        if 'variables' in req:
            template['variables'] = req['variables']

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


@chisel.action(name='getModels', types=OLLAMA_CHAT_TYPES)
def get_models(ctx, unused_req):
    # Get the current model
    with ctx.app.config() as config:
        current_model = ConfigManager.get_model(config)

    # Get the Ollama models
    try:
        models = ollama.list()['models'] or ()
    except: # pylint: disable=bare-except
        models = ()

    # Create the models response
    response_models = [
        {
            'id': model.model,
            'name': model.model[:model.model.index(':')],
            'parameters': _parse_parameter_size(model.details.parameter_size),
            'size': model.size,
            'modified': model.modified_at
        }
        for model in models
    ]

    # Create the downloading models response
    downloading_models = []
    for model_id, download_manager in ctx.app.downloads.items():
        download = {
            'id': model_id,
            'status': download_manager.status,
            'completed': download_manager.completed
        }
        if download_manager.total:
            download['size'] = download_manager.total
        downloading_models.append(download)

    return {
        'model': current_model,
        'models': sorted(response_models, key=lambda model: model['id']),
        'downloading': sorted(downloading_models, key=lambda model: model['id'])
    }


# Helper function to parse parameter sizes
def _parse_parameter_size(parameter_size):
    value = float(parameter_size[:-1])
    unit = parameter_size[-1]
    if unit == 'B':
        return int(value * 1000000000)
    elif unit == 'M':
        return int(value * 1000000)
    elif unit == 'K':
        return int(value * 1000)
    raise ValueError(f'Unrecognized parameter size: {parameter_size}')


@chisel.action(name='downloadModel', types=OLLAMA_CHAT_TYPES)
def download_model(ctx, req):
    with ctx.app.config():
        ctx.app.downloads[req['model']] = DownloadManager(ctx.app, req['model'])


@chisel.action(name='stopModelDownload', types=OLLAMA_CHAT_TYPES)
def stop_model_download(ctx, req):
    with ctx.app.config():
        if req['model'] in ctx.app.downloads:
            ctx.app.downloads[req['model']].stop = True


@chisel.action(name='deleteModel', types=OLLAMA_CHAT_TYPES)
def delete_model(unused_ctx, req):
    ollama.delete(req['model'])


@chisel.action(name='getSystemInfo', types=OLLAMA_CHAT_TYPES)
def get_system_info(unused_ctx, unused_req):
    # Compute the total memory
    if os.name == 'nt':
        memory_status = MEMORYSTATUSEX()
        # pylint: disable-next=invalid-name, attribute-defined-outside-init
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        total_memory = memory_status.ullTotalPhys
    else:
        # pylint: disable-next=no-member, useless-suppression
        total_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    return {
        'memory': total_memory
    }


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint),
        ("dwMemoryLoad", ctypes.c_uint),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]
