# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat back-end application
"""

import importlib.resources as pkg_resources

import chisel

from .ollama import OllamaChat


# The default Ollama chat model
DEFAULT_MODEL = 'phi3'


class OllamaChatApplication(chisel.Application):
    """
    The ollama-chat back-end API WSGI application class
    """

    __slots__ = ('chats',)


    def __init__(self):
        super().__init__()

        # The application's chats
        self.chats = {}

        # Add the chisel documentation application
        self.add_requests(chisel.create_doc_requests())

        # Add the ollama-chat APIs
        self.add_request(start_chat)
        self.add_request(get_chat)
        self.add_request(stop_chat)

        # Add the ollama-chat statics
        self.add_static(
            'index.html',
            'text/html; charset=utf-8',
            (('GET', None), ('GET', '/')),
            'The Ollama Chat application HTML'
        )
        self.add_static(
            'ollamaChat.mds',
            'text/plain; charset=utf-8',
            (('GET', None),),
            'The Ollama Chat application markdown-script'
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


@chisel.action(spec='''\
group "Ollama Chat API"


# Start a language model chat
action start_chat
    urls
        POST

    input
        # The language model name. If not provided, use the default model.
        optional string model

        # The chat prompt
        string prompt

    output
        # The chat ID
        int id
''')
def start_chat(ctx, req):
    chat_id = len(ctx.app.chats)
    chat = OllamaChat(req.get('model', DEFAULT_MODEL), req['prompt'])
    ctx.app.chats[chat_id] = chat
    return {
        'id': chat_id
    }


@chisel.action(spec='''\
# Get the language model chat text
action get_chat
    urls
        GET

    query
        # The chat ID
        int id

    output
        # The language model name
        string model

        # The chat prompt
        string prompt

        # The chat response
        string response

        # If True, the chat is completed
        bool completed

    errors
        UnknownChatID
''')
def get_chat(ctx, req):
    chat_id = req['id']
    if chat_id not in ctx.app.chats:
        raise chisel.ActionError('UnknownChatID')
    chat = ctx.app.chats[chat_id]
    response, completed = chat.get_response()
    return {
        'model': chat.model,
        'prompt': chat.prompt,
        'response': response,
        'completed': completed
    }


@chisel.action(spec='''\
# Stop a language model chat
action stop_chat
    urls
        POST

    input
        # The chat ID
        int id

    errors
        UnknownChatID
''')
def stop_chat(ctx, req):
    chat_id = req['id']
    if chat_id not in ctx.app.chats:
        raise chisel.ActionError('UnknownChatID')
    chat = ctx.app.chats[chat_id]
    chat.stop()
