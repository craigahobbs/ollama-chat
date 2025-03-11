# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
ollama-chat command-line script main module
"""

import argparse
import json
import os
import threading
import urllib.request
import webbrowser

from schema_markdown import encode_query_string
import waitress

from .app import OllamaChat


# The default config file name
CONFIG_FILENAME = 'ollama-chat.json'


def main(argv=None):
    """
    ollama-chat command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(prog='ollama-chat')
    parser.add_argument('-c', metavar='FILE', dest='config',
                        help='the configuration file (default is "$HOME/ollama-chat.json")')
    parser.add_argument('-m', metavar='MESSAGE', dest='message',
                        help='start a conversation')
    parser.add_argument('-t', metavar='TEMPLATE', dest='template',
                        help='start a template')
    parser.add_argument('-l', metavar='MODEL', dest='model',
                        help='the model name (default is current model)')
    parser.add_argument('-v', nargs=2, action='append', metavar=('VAR', 'VALUE'), dest='template_vars', default = [],
                        help='the template variables')
    parser.add_argument('-p', metavar='N', dest='port', type=int, default=8080,
                        help='the application port (default is 8080)')
    parser.add_argument('-x', dest='xorigin', action='store_true', default=False,
                        help="enable cross-origin back-end requests")
    parser.add_argument('-b', dest='backend', action='store_false', default=True,
                        help="don't start the back-end (use existing)")
    parser.add_argument('-n', dest='browser', action='store_false', default=True,
                        help="don't open a web browser")
    parser.add_argument('-q', dest='quiet', action='store_true',
                        help="don't display access logging")
    args = parser.parse_args(args=argv)

    # Starting a backend server? If so, create the backend application.
    if args.backend:

        # Determine the config path
        config_path = args.config
        if config_path is None:
            if os.path.isfile(CONFIG_FILENAME):
                config_path = CONFIG_FILENAME
            else:
                config_path = os.path.join(os.path.expanduser('~'), CONFIG_FILENAME)
        elif config_path.endswith(os.sep) or os.path.isdir(config_path):
            config_path = os.path.join(config_path, CONFIG_FILENAME)

        # Create the backend application
        application = OllamaChat(config_path, args.xorigin)

    # Construct the URL
    host = '127.0.0.1'
    url = f'http://{host}:{args.port}/'
    browser_url = url

    # Conversation command?
    if args.message:

        # Start the conversation
        request_input = {'user': args.message}
        if args.model:
            request_input['model'] = args.model
        request_bytes = json.dumps(request_input).encode('utf-8')
        if args.backend:
            _, _, response_bytes = application.request('POST', '/startConversation', wsgi_input=request_bytes)
        else:
            request = urllib.request.Request(f'{url}startConversation', data=request_bytes)
            with urllib.request.urlopen(request) as response:
                response_bytes = response.read()
        response = json.loads(response_bytes.decode('utf-8'))

        # Update the browser URL
        message_args = encode_query_string({'var': {'vView': "'chat'", 'vId': f"'{response['id']}'"}})
        browser_url = f'{url}#{message_args}&chat-bottom'

    # Template command?
    elif args.template:

        # Start the template
        request_input = {'id': args.template, 'variables': dict(args.template_vars)}
        if args.model:
            request_input['model'] = args.model
        request_bytes = json.dumps(request_input).encode('utf-8')
        if args.backend:
            _, _, response_bytes = application.request('POST', '/startTemplate', wsgi_input=request_bytes)
        else:
            try:
                request = urllib.request.Request(f'{url}startTemplate', data=request_bytes)
                with urllib.request.urlopen(request) as response:
                    response_bytes = response.read()
            except urllib.request.HTTPError as exc:
                response_bytes = exc.fp.read()
        response = json.loads(response_bytes.decode('utf-8'))
        if 'error' in response:
            parser.error(response.get('message') or response["error"])

        # Update the browser URL
        template_args = encode_query_string({'var': {'vView': "'chat'", 'vId': f"'{response['id']}'"}})
        browser_url = f'{url}#{template_args}&chat-bottom'

    # Launch the web browser on a thread (it may block)
    if args.browser:
        webbrowser_thread = threading.Thread(target=webbrowser.open, args=(browser_url,))
        webbrowser_thread.daemon = True
        webbrowser_thread.start()

    # Host the application
    if args.backend:

        # Wrap the backend so we can log status and environ
        def application_wrap(environ, start_response):
            def log_start_response(status, response_headers):
                if not args.quiet:
                    print(f'ollama-chat: {status[0:3]} {environ["REQUEST_METHOD"]} {environ["PATH_INFO"]} {environ["QUERY_STRING"]}')
                return start_response(status, response_headers)
            return application(environ, log_start_response)

        # Start the backend application
        if not args.quiet:
            print(f'ollama-chat: Serving at {url} ...')
        waitress.serve(application_wrap, port=args.port)

    # Not starting a backend service, so we must wait on the web browser start
    elif args.browser:
        webbrowser_thread.join()
