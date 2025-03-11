# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

from contextlib import contextmanager
from io import StringIO
import json
import os
from tempfile import TemporaryDirectory
import unittest
import unittest.mock

import chisel
from ollama_chat.__main__ import main as main_main
from ollama_chat.main import main


# Helper context manager to create a list of files in a temporary directory
@contextmanager
def create_test_files(file_defs):
    tempdir = TemporaryDirectory()
    try:
        for path_parts, content in file_defs:
            path = os.path.join(tempdir.name, *path_parts)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as file_:
                file_.write(content)
        yield tempdir.name
    finally:
        tempdir.cleanup()


class TestMain(unittest.TestCase):

    def test_main_submodule(self):
        self.assertIs(main_main, main)


    def test_main_help(self):
        with unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-h'])

            self.assertEqual(cm_exc.exception.code, 0)
            self.assertTrue(stdout.getvalue().startswith('usage: ollama-chat [-h] [-c FILE] [-m MESSAGE] [-t TEMPLATE] [-l MODEL]'))
            self.assertEqual(stderr.getvalue(), '')


    def test_main_config_default(self):
        with unittest.mock.patch('os.path.isfile', return_value=False) as mock_isfile, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main([])

            self.assertEqual(mock_isfile.call_count, 2)
            self.assertTupleEqual(mock_isfile.call_args_list[0].args, ('ollama-chat.json',))
            self.assertTupleEqual(mock_isfile.call_args_list[1].args, (os.path.join(os.path.expanduser('~'), 'ollama-chat.json'),))

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})

            self.assertEqual(
                stdout.getvalue(),
                '''\
ollama-chat: Serving at http://127.0.0.1:8080/ ...
ollama-chat: 200 GET /getConversations\x20
'''
            )
            self.assertEqual(stderr.getvalue(), '')


    def test_main_config_cwd(self):
        with unittest.mock.patch('os.path.isfile', return_value=True) as mock_isfile, \
             unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"model": "llm", "conversations": []}')), \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main([])

            self.assertEqual(mock_isfile.call_count, 2)
            self.assertTupleEqual(mock_isfile.call_args_list[0].args, ('ollama-chat.json',))
            self.assertTupleEqual(mock_isfile.call_args_list[1].args, ('ollama-chat.json',))

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})

            self.assertEqual(
                stdout.getvalue(),
                '''\
ollama-chat: Serving at http://127.0.0.1:8080/ ...
ollama-chat: 200 GET /getConversations\x20
'''
            )
            self.assertEqual(stderr.getvalue(), '')


    def test_main_config_dir_default(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', input_dir])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})

            self.assertEqual(
                stdout.getvalue(),
                '''\
ollama-chat: Serving at http://127.0.0.1:8080/ ...
ollama-chat: 200 GET /getConversations\x20
'''
            )
            self.assertEqual(stderr.getvalue(), '')


    def test_main_config_dir_exist(self):
        with create_test_files([
            (('ollama-chat.json',), '{"model": "llm", "conversations": []}')
        ]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', input_dir])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})

            self.assertEqual(
                stdout.getvalue(),
                '''\
ollama-chat: Serving at http://127.0.0.1:8080/ ...
ollama-chat: 200 GET /getConversations\x20
'''
            )
            self.assertEqual(stderr.getvalue(), '')


    def test_main_config_file(self):
        with create_test_files([
            (('ollama-chat.json',), '{"model": "llm", "conversations": []}')
        ]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', os.path.join(input_dir, 'ollama-chat.json')])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})

            self.assertEqual(
                stdout.getvalue(),
                '''\
ollama-chat: Serving at http://127.0.0.1:8080/ ...
ollama-chat: 200 GET /getConversations\x20
'''
            )
            self.assertEqual(stderr.getvalue(), '')


    def test_main_quiet(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-q', '-c', input_dir])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            application_wrap = serve_args[0]
            self.assertTrue(callable(application_wrap))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            start_response_calls = []
            def start_response(status, response_headers):
                start_response_calls.append((status, response_headers))
            environ = chisel.Context.create_environ('GET', '/getConversations')
            response = json.loads(application_wrap(environ, start_response)[0].decode('utf-8'))

            self.assertListEqual(start_response_calls, [('200 OK', [('Content-Type', 'application/json')])])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_backend(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-c', input_dir])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_called_once_with()

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_browser(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-n', '-c', input_dir])

            mock_thread.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            self.assertTrue(callable(serve_args[0]))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_backend_no_browser(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-n', '-c', input_dir])

            mock_thread.assert_not_called()

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_conversation(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'conv1'}).encode('utf-8'))

            main(['-c', input_dir, '-m', 'Hello'])

            mock_application.request.assert_called_once_with(
                'POST', '/startConversation', wsgi_input=json.dumps({'user': 'Hello'}).encode('utf-8')
            )

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27conv1%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_conversation_with_model(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'conv2'}).encode('utf-8'))

            main(['-c', input_dir, '-m', 'Hello', '-l', 'llm'])

            mock_application.request.assert_called_once_with(
                'POST', '/startConversation', wsgi_input=json.dumps({'user': 'Hello', 'model': 'llm'}).encode('utf-8')
            )

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27conv2%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_conversation_no_backend(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('urllib.request.urlopen') as mock_urlopen, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            # Set up the mock response object
            mock_response_obj = unittest.mock.MagicMock()
            mock_response_obj.read.return_value = json.dumps({'id': 'conv3'}).encode('utf-8')

            # Set up the context manager mock
            mock_response = unittest.mock.MagicMock()
            mock_response.__enter__.return_value = mock_response_obj
            mock_urlopen.return_value = mock_response

            main(['-c', input_dir, '-b', '-m', 'Hello'])

            mock_urlopen.assert_called_once()
            request = mock_urlopen.call_args[0][0]
            self.assertEqual(request.full_url, 'http://127.0.0.1:8080/startConversation')
            self.assertEqual(request.data, json.dumps({'user': 'Hello'}).encode('utf-8'))

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27conv3%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_called_once_with()

            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')
