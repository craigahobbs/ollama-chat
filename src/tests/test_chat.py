# {% raw %}
# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import base64
import json
import os
import unittest
import unittest.mock

from ollama_chat.app import OllamaChat
from ollama_chat.chat import _escape_markdown_text, _process_commands, config_template_prompts, ChatManager

from .util import create_test_files


class TestChatManaper(unittest.TestCase):

    def test_chat_fn(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Configure the ollama.chat mock
            mock_chunks = [['Hi ', 'there!'], ['Bye ', 'bye!']]
            mock_chat.side_effect = [iter({'message': {'content': chunk}} for chunk in chunks) for chunks in mock_chunks]

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            app.chats['conv1'] = None
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {
                                    'user': 'Hello',
                                    'model': 'Hi there!'
                                },
                                {
                                    'user': 'Goodbye',
                                    'model': 'Bye bye!'
                                }
                            ]
                        }
                    ]
                })


    def test_chat_fn_help(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Create the ChatManager instance
            chat_prompts = ['/?']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            mock_chat.assert_not_called()
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                exchange = config['conversations'][0]['exchanges'][0]
                self.assertTrue(exchange['model'].startswith('```\nusage: /{?,dir,do,file,image,url}'))
                del exchange['model']
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {
                                    'user': '/?'
                                }
                            ]
                        }
                    ]
                })


    def test_chat_fn_show(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            })),
            (('test.txt',), 'file content')
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Create the ChatManager instance
            chat_prompts = [f'This file:\n\n/file {temp_dir}/test.txt -n']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            mock_chat.assert_not_called()
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {
                                    'user': f'''\
This file:

/file {temp_dir}/test.txt -n''',
                                    'model': f'''\
This file:

<{_escape_markdown_text(temp_dir)}/test.txt>
file content
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
                                }
                            ]
                        }
                    ]
                })


    def test_chat_fn_do(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ],
                'templates': [
                    {'id': 'tmpl1', 'name': 'bye', 'title': 'Goodbye', 'prompts': ['Goodbye']}
                ]
            }))
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Configure the ollama.chat mock
            mock_chunks = [['Bye ', 'bye!']]
            mock_chat.side_effect = [iter({'message': {'content': chunk}} for chunk in chunks) for chunks in mock_chunks]

            # Create the ChatManager instance
            chat_prompts = ['/do bye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            app.chats['conv1'] = None
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {'user': '/do bye', 'model': 'Executing template "bye"'},
                                {'user': 'Goodbye', 'model': 'Bye bye!'}
                            ]
                        }
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'bye', 'title': 'Goodbye', 'prompts': ['Goodbye']}
                    ]
                })


    def test_chat_fn_do_variables(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ],
                'templates': [
                    {
                        'id': 'tmpl1',
                        'name': 'bye',
                        'title': 'Goodbye',
                        'variables': [{'name': 'name', 'label': 'Name'}],
                        'prompts': ['Goodbye, {{name}}']
                    }
                ]
            }))
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Configure the ollama.chat mock
            mock_chunks = [['Bye ', 'bye!']]
            mock_chat.side_effect = [iter({'message': {'content': chunk}} for chunk in chunks) for chunks in mock_chunks]

            # Create the ChatManager instance
            chat_prompts = ['/do bye -v name Joe']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            app.chats['conv1'] = None
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {'user': '/do bye -v name Joe', 'model': 'Executing template "bye" - name = "Joe"'},
                                {'user': 'Goodbye, Joe', 'model': 'Bye bye!'}
                            ]
                        }
                    ],
                    'templates': [
                        {
                            'id': 'tmpl1',
                            'name': 'bye',
                            'title': 'Goodbye',
                            'variables': [{'name': 'name', 'label': 'Name'}],
                            'prompts': ['Goodbye, {{name}}']
                        }
                    ]
                })


    def test_chat_fn_do_unknown_template(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir, \
        unittest.mock.patch('threading.Thread') as mock_thread, \
        unittest.mock.patch('ollama.chat') as mock_chat:
            # Create the ChatManager instance
            chat_prompts = ['/do unknown']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager,))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager)
            mock_chat.assert_not_called()
            self.assertDictEqual(app.chats, {})
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {
                                    'user': '/do unknown',
                                    'model': '\n**ERROR:** unknown template "unknown"'
                                }
                            ]
                        }
                    ]
                })


class TestConfigTemplatePrompts(unittest.TestCase):

    def test_basic(self):
        template = {
            'title': 'Simple Template',
            'prompts': ['Hello world', 'Second prompt']
        }
        variable_values = {}
        title, prompts = config_template_prompts(template, variable_values)
        self.assertEqual(title, 'Simple Template')
        self.assertListEqual(prompts, ['Hello world', 'Second prompt'])


    def test_variables(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}', 'How are you {{name}}?'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {'name': 'Alice'}
        title, prompts = config_template_prompts(template, variable_values)
        self.assertEqual(title, 'Hello Alice')
        self.assertListEqual(prompts, ['Greetings Alice', 'How are you Alice?'])


    def test_missing_variable(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {}
        with self.assertRaises(ValueError) as context:
            config_template_prompts(template, variable_values)
        self.assertEqual(str(context.exception), 'missing variable value for "name"')


    def test_unknown_variable(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {'name': 'Alice', 'age': '30'}
        with self.assertRaises(ValueError) as context:
            config_template_prompts(template, variable_values)
        self.assertEqual(str(context.exception), 'unknown variable "age"')


    def test_multiple_variables(self):
        template = {
            'title': '{{greeting}} {{name}}',
            'prompts': ['{{greeting}} dear {{name}}', '{{name}}, how are you?'],
            'variables': [{'name': 'greeting'}, {'name': 'name'}]
        }
        variable_values = {'greeting': 'Hello', 'name': 'Bob'}
        title, prompts = config_template_prompts(template, variable_values)
        self.assertEqual(title, 'Hello Bob')
        self.assertListEqual(prompts, ['Hello dear Bob', 'Bob, how are you?'])


class TestProcessCommands(unittest.TestCase):

    def test_no_commands(self):
        flags = {}
        self.assertEqual(_process_commands('Hello, how are you?', flags), 'Hello, how are you?')
        self.assertDictEqual(flags, {})


    def test_help_top(self):
        flags = {}
        self.assertEqual(_process_commands('/?', flags), 'Displaying top-level help')
        self.assertTrue(flags['help'].startswith('usage: /{?,dir,do,file,image,url}'))


    def test_help(self):
        flags = {}
        self.assertEqual(_process_commands('/file test.txt -h', flags), 'Displaying help for "file" command')
        self.assertTrue(flags['help'].startswith('usage: /file [-h] [-n] file'))


    def test_dir(self):
        with create_test_files([
            (('test.txt',), 'Test 1'),
            (('subdir', 'test2.txt',), 'Test 2'),
            (('subdir', 'test3.md',), '# Test 3')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/dir {temp_dir} .txt', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/test.txt>
Test 1
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_dir_depth(self):
        with create_test_files([
            (('test.txt',), 'Test 1'),
            (('subdir', 'test2.txt',), 'Test 2'),
            (('subdir', 'test3.md',), '# Test 3')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/dir {temp_dir} .txt -d 2', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/subdir/test2.txt>
Test 2
</ {_escape_markdown_text(temp_dir)}/subdir/test2.txt>

<{_escape_markdown_text(temp_dir)}/test.txt>
Test 1
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_do(self):
        flags = {}
        self.assertEqual(_process_commands('/do template_name -v var1 val1', flags), 'Executing template "template_name"')
        self.assertDictEqual(flags, {'do': [('template_name', {'var1': 'val1'})]})


    def test_file(self):
        with create_test_files([
            (('test.txt',), 'file content')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_dir}/test.txt', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/test.txt>
file content
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_file_show(self):
        with create_test_files([
            (('test.txt',), 'file content')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_dir}/test.txt -n', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/test.txt>
file content
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {'show': True})


    def test_image(self):
        with create_test_files([
            (('test.jpg',), 'image data')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(_process_commands(f'/image {temp_dir}/test.jpg', flags), '')
            self.assertDictEqual(flags, {'images': [base64.b64encode(b'image data').decode('utf-8')]})


    def test_url(self):
        with unittest.mock.patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = unittest.mock.Mock()
            mock_response.read.return_value = b'url content'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            flags = {}
            self.assertEqual(
                _process_commands('/url http://example.com', flags),
                '''\
<http://example.com>
url content
</ http://example.com>'''
            )
            self.assertDictEqual(flags, {})


    def test_multiple_commands(self):
        with create_test_files([
            (('test.txt',), 'file content')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'Hello\n\n/?\n\n/file {temp_dir}/test.txt', flags),
                f'''\
Hello

Displaying top-level help

<{_escape_markdown_text(temp_dir)}/test.txt>
file content
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertIn('help', flags)


    def test_file_error(self):
        with create_test_files([]) as temp_dir:
            flags = {}
            with self.assertRaises(FileNotFoundError):
                _process_commands(f'/file {temp_dir}/nonexistent.txt', flags)

# {% endraw %}
