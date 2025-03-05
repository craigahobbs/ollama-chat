# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

from contextlib import contextmanager
from io import StringIO
import os
from tempfile import TemporaryDirectory
import unittest
import unittest.mock

from ollama_chat.__main__ import main as main_main
from ollama_chat.main import main


# Helper context manager to create a list of files in a temporary directory
@contextmanager
def create_test_files(file_defs):
    tempdir = TemporaryDirectory()
    try:
        for path_parts, content in file_defs:
            if isinstance(path_parts, str):
                path_parts = [path_parts]
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


    def test_main_default(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-c', input_dir])

            mock_open.assert_called_once_with('http://127.0.0.1:8080/')

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            self.assertTrue(callable(serve_args[0]))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_quiet(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-q', '-c', input_dir])

            mock_open.assert_called_once_with('http://127.0.0.1:8080/')

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            self.assertTrue(callable(serve_args[0]))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_backend(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-c', input_dir])

            mock_open.assert_called_once_with('http://127.0.0.1:8080/')

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_browser(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-n', '-c', input_dir])

            mock_open.assert_not_called()

            mock_serve.assert_called_once()
            serve_args, serve_kwargs = mock_serve.call_args
            self.assertTrue(callable(serve_args[0]))
            self.assertDictEqual(serve_kwargs, {'port': 8080})

            self.assertEqual(stdout.getvalue(), 'ollama-chat: Serving at http://127.0.0.1:8080/ ...\n')
            self.assertEqual(stderr.getvalue(), '')


    def test_main_no_backend_no_browser(self):
        with create_test_files([]) as input_dir, \
             unittest.mock.patch('waitress.serve') as mock_serve, \
             unittest.mock.patch('webbrowser.open') as mock_open, \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:

            main(['-b', '-n', '-c', input_dir])

            mock_open.assert_not_called()

            mock_serve.assert_not_called()

            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(stderr.getvalue(), '')
