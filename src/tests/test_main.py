# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

from io import StringIO
import json
import os
import unittest
import unittest.mock

import chisel
import urllib3
from ollama_chat.__main__ import main as main_main
from ollama_chat.main import main

from .util import create_test_files


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
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', temp_dir])

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
        test_files = [
            ('ollama-chat.json', '{"model": "llm", "conversations": []}')
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', temp_dir])

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
        test_files = [
            ('ollama-chat.json', '{"model": "llm", "conversations": []}')
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', os.path.join(temp_dir, 'ollama-chat.json')])

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
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-q', '-c', temp_dir])

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
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-c', temp_dir])

            mock_thread.assert_called_once_with(target=mock_open, args=('http://127.0.0.1:8080/',))
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_called_once_with()

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_browser(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-n', '-c', temp_dir])

            mock_thread.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            self.assertTrue(callable(serve_args[0]))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_backend_no_browser(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-n', '-c', temp_dir])

            mock_thread.assert_not_called()

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_conversation(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'conv1'}).encode('utf-8'))

            main(['-c', temp_dir, '-m', 'Hello'])

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
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'conv2'}).encode('utf-8'))

            main(['-c', temp_dir, '-m', 'Hello', '-l', 'llm'])

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
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            # Set up the mock response object
            mock_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_response.status = 200
            mock_response.json.return_value = {'id': 'conv3'}
            mock_request.return_value = mock_response

            main(['-c', temp_dir, '-b', '-m', 'Hello'])

            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startConversation', json={'user': 'Hello'}, retries=0
            )
            mock_response.close.assert_called_once_with()

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


    def test_main_start_conversation_error(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            # Set up the mock response object
            mock_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_response.status = 500
            mock_response.json.return_value = {'error': 'UnexpectedError'}
            mock_request.return_value = mock_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', temp_dir, '-b', '-m', 'Hello'])

            self.assertEqual(cm_exc.exception.code, 2)
            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startConversation', json={'user': 'Hello'}, retries=0
            )
            mock_response.close.assert_called_once_with()

            mock_thread.assert_not_called()
            mock_open.assert_not_called()
            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertTrue(stderr.getvalue().endswith('ollama-chat: error: Failed to start conversation\n'))


    def test_main_start_conversation_error_unexpected(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            def mock_request_side_effect(*args, **kwargs):
                raise Exception('BAD')
            mock_request.side_effect = mock_request_side_effect

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', temp_dir, '-b', '-m', 'Hello'])

            self.assertEqual(cm_exc.exception.code, 2)
            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startConversation', json={'user': 'Hello'}, retries=0
            )

            mock_thread.assert_not_called()
            mock_open.assert_not_called()
            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertTrue(stderr.getvalue().endswith('ollama-chat: error: Failed to start conversation\n'))


    def test_main_start_template(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'temp1'}).encode('utf-8'))

            main(['-c', temp_dir, '-t', 'template1'])

            mock_application.request.assert_called_once_with(
                'POST', '/startTemplate', wsgi_input=json.dumps({'id': 'template1', 'variables': {}}).encode('utf-8')
            )

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27temp1%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_template_with_model(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'temp2'}).encode('utf-8'))

            main(['-c', temp_dir, '-t', 'template1', '-l', 'llm'])

            mock_application.request.assert_called_once_with(
                'POST', '/startTemplate',
                wsgi_input=json.dumps({'id': 'template1', 'variables': {}, 'model': 'llm'}).encode('utf-8')
            )

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27temp2%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_template_with_vars(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.main.OllamaChat') as mock_ollama_chat, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_application = mock_ollama_chat.return_value
            mock_application.request.return_value = (200, [], json.dumps({'id': 'temp3'}).encode('utf-8'))

            main(['-c', temp_dir, '-t', 'template1', '-v', 'var1', 'val1', '-v', 'var2', 'val2'])

            mock_application.request.assert_called_once_with(
                'POST', '/startTemplate',
                wsgi_input=json.dumps({'id': 'template1', 'variables': {'var1': 'val1', 'var2': 'val2'}}).encode('utf-8')
            )

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27temp3%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_not_called()

            mock_serve.assert_called_once()
            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_template_no_backend(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_response.status = 200
            mock_response.json.return_value = {'id': 'temp4'}
            mock_request.return_value = mock_response

            main(['-c', temp_dir, '-b', '-t', 'template1'])

            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startTemplate', json={'id': 'template1', 'variables': {}}, retries=0
            )
            mock_response.close.assert_called_once_with()

            mock_thread.assert_called_once_with(
                target=mock_open,
                args=('http://127.0.0.1:8080/#var.vId=%27temp4%27&var.vView=%27chat%27&chat-bottom',)
            )
            thread_instance = mock_thread.return_value
            self.assertTrue(thread_instance.daemon)
            thread_instance.start.assert_called_once_with()
            thread_instance.join.assert_called_once_with()

            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_start_template_error(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            mock_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_response.status = 400
            mock_response.json.return_value = {'error': 'UnexpectedError', 'message': 'Invalid template'}
            mock_request.return_value = mock_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', temp_dir, '-b', '-t', 'bad_template'])

            self.assertEqual(cm_exc.exception.code, 2)
            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startTemplate', json={'id': 'bad_template', 'variables': {}}, retries=0
            )
            mock_response.close.assert_called_once_with()
            mock_thread.assert_not_called()

            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertTrue(stderr.getvalue().endswith('ollama-chat: error: Failed to start template "bad_template"\n'))


    def test_main_start_template_error_unexpected(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.request') as mock_request, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            def mock_request_side_effect(*args, **kwargs):
                raise Exception('BAD')
            mock_request.side_effect = mock_request_side_effect

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', temp_dir, '-b', '-t', 'bad_template'])

            self.assertEqual(cm_exc.exception.code, 2)
            mock_request.assert_called_once_with(
                'POST', 'http://127.0.0.1:8080/startTemplate', json={'id': 'bad_template', 'variables': {}}, retries=0
            )
            mock_thread.assert_not_called()

            mock_serve.assert_not_called()
            self.assertEqual(stdout.getvalue(), '')
            self.assertTrue(stderr.getvalue().endswith('ollama-chat: error: Failed to start template "bad_template"\n'))
