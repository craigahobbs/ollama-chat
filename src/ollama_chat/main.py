# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
ollama-chat command-line script main module
"""

import argparse
import threading
import webbrowser

import waitress

from .app import OllamaChatApplication


def main(argv=None):
    """
    ollama-chat command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(prog='ollama-chat')
    parser.add_argument('-p', metavar='N', dest='port', type=int, default=8080,
                        help='the application port (default is 8080)')
    parser.add_argument('-n', dest='no_browser', action='store_true',
                        help="don't open a web browser")
    parser.add_argument('-q', dest='quiet', action='store_true',
                        help="don't display access logging")
    args = parser.parse_args(args=argv)

    # Construct the URL
    host = '127.0.0.1'
    url = f'http://{host}:{args.port}/'

    # Launch the web browser on a thread so the WSGI application can startup first
    if not args.no_browser:
        webbrowser_thread = threading.Thread(target=webbrowser.open, args=(url,))
        webbrowser_thread.daemon = True
        webbrowser_thread.start()

    # Create the WSGI application
    wsgiapp = OllamaChatApplication()

    # Wrap the WSGI application and the start_response function so we can log status and environ
    def wsgiapp_wrap(environ, start_response):
        def log_start_response(status, response_headers):
            if not args.quiet:
                print(f'ollama-chat: {status[0:3]} {environ["REQUEST_METHOD"]} {environ["PATH_INFO"]} {environ["QUERY_STRING"]}')
            return start_response(status, response_headers)
        return wsgiapp(environ, log_start_response)

    # Host the application
    if not args.quiet:
        print(f'ollama-chat: Serving at {url} ...')
    waitress.serve(wsgiapp_wrap, port=args.port)
