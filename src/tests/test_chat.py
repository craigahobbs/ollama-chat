# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import base64
import json
import os
import pathlib
import unittest
import unittest.mock

import urllib3

from ollama_chat.app import OllamaChat
from ollama_chat.chat import _escape_markdown_text, _process_commands, config_template_prompts, ChatManager

from .util import create_test_files


class TestChatManager(unittest.TestCase):

    def test_chat_fn(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Create a mock show response
            mock_show_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response2.status = 200
            mock_show_response2.json.return_value = {'capabilities': []}

            # Create a second mock chat response
            mock_chat_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response2.status = 200
            mock_chat_response2.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Bye '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'bye!'}}).encode('utf-8')
            ]

            # Configure the mock pool manager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [
                mock_show_response, mock_chat_response, mock_show_response2, mock_chat_response2
            ]

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the ollama.chat calls
            self.assertEqual(mock_pool_manager_instance.request.call_count, 4)
            self.assertListEqual(
                mock_pool_manager_instance.request.call_args_list,
                [
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'}),
                    unittest.mock.call(
                        'POST',
                        'http://127.0.0.1:11434/api/chat',
                        json={
                            'model': 'llm',
                            'messages': [
                                {'role': 'user', 'content': 'Hello', 'images': None}
                            ],
                            'stream': True,
                            'think': False
                        },
                        preload_content=False
                    ),
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'}),
                    unittest.mock.call(
                        'POST',
                        'http://127.0.0.1:11434/api/chat',
                        json={
                            'model': 'llm',
                            'messages': [
                                {'role': 'user', 'content': 'Hello', 'images': None},
                                {'role': 'assistant', 'content': 'Hi there!'},
                                {'role': 'user', 'content': 'Goodbye', 'images': None}
                            ],
                            'stream': True,
                            'think': False
                        },
                        preload_content=False
                    )
                ]
            )

            # Verify the app config
            expected_config = {
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
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)

    def test_chat_fn_thinking(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': ['thinking']}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'thinking': 'Hmmm '}}).encode('utf-8'),
                json.dumps({'message': {'thinking': 'Haw'}}).encode('utf-8'),
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Create a second mock show response
            mock_show_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response2.status = 200
            mock_show_response2.json.return_value = {'capabilities': ['thinking']}

            # Create a second mock chat response
            mock_chat_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response2.status = 200
            mock_chat_response2.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Bye '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'bye!'}}).encode('utf-8')
            ]

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [
                mock_show_response, mock_chat_response, mock_show_response2, mock_chat_response2
            ]

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the ollama.chat calls
            self.assertListEqual(
                mock_pool_manager_instance.request.call_args_list,
                [
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'}),
                    unittest.mock.call(
                        'POST',
                        'http://127.0.0.1:11434/api/chat',
                        json={
                            'model': 'llm',
                            'messages': [
                                {'role': 'user', 'content': 'Hello', 'images': None}
                            ],
                            'stream': True,
                            'think': True
                        },
                        preload_content=False
                    ),
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'}),
                    unittest.mock.call(
                        'POST',
                        'http://127.0.0.1:11434/api/chat',
                        json={
                            'model': 'llm',
                            'messages': [
                                {'role': 'user', 'content': 'Hello', 'images': None},
                                {'role': 'assistant', 'content': 'Hi there!'},
                                {'role': 'user', 'content': 'Goodbye', 'images': None}
                            ],
                            'stream': True,
                            'think': True
                        },
                        preload_content=False
                    )
                ]
            )

            # Verify the app config
            expected_config = {
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {
                                'user': 'Hello',
                                'model': 'Hi there!',
                                'thinking': 'Hmmm Haw'
                            },
                            {
                                'user': 'Goodbye',
                                'model': 'Bye bye!'
                            }
                        ]
                    }
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_error_show(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 500

            # Configure the mock pool manager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_show_response

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the ollama.chat calls
            self.assertEqual(mock_pool_manager_instance.request.call_count, 1)
            self.assertListEqual(
                mock_pool_manager_instance.request.call_args_list,
                [
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'})
                ]
            )

            # Verify the app config
            expected_config = {
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {
                                'user': 'Hello',
                                'model': '\n**ERROR:** Unknown model "llm" (500)'
                            }
                        ]
                    }
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_error_chat(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': ['thinking']}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 500

            # Configure the mock pool manager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the ollama.chat calls
            self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
            self.assertListEqual(
                mock_pool_manager_instance.request.call_args_list,
                [
                    unittest.mock.call('POST', 'http://127.0.0.1:11434/api/show', json={'model': 'llm'}),
                    unittest.mock.call(
                        'POST',
                        'http://127.0.0.1:11434/api/chat',
                        json={
                            'model': 'llm',
                            'messages': [{'role': 'user', 'content': 'Hello', 'images': None}],
                            'stream': True,
                            'think': True
                        },
                        preload_content=False
                    )
                ]
            )

            # Verify the app config
            expected_config = {
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {
                                'user': 'Hello',
                                'model': '\n**ERROR:** Unknown model "llm" (500)'
                            }
                        ]
                    }
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_stop(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Create a second mock show response
            mock_show_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response2.status = 200
            mock_show_response2.json.return_value = {'capabilities': []}

            # Create a seconde mock chat response
            mock_chat_response2 = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response2.status = 200
            mock_chat_response2.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Bye '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'bye!'}}).encode('utf-8')
            ]

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [
                mock_show_response, mock_chat_response, mock_show_response2, mock_chat_response2
            ]

            # Create the ChatManager instance
            chat_prompts = ['Hello', 'Goodbye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            chat_manager.stop = True
            # app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello','model': ''}]}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_help(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value

            # Create the ChatManager instance
            chat_prompts = ['/?']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            mock_pool_manager_instance.request.assert_not_called()
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': '/?'}]}
                ]
            }
            with app.config() as config:
                exchange = config['conversations'][0]['exchanges'][0]
                self.assertTrue(exchange['model'].startswith('```\nusage: /{?,dir,do,file,image,url}'))
                del exchange['model']
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                config = json.load(config_fh)
                exchange = config['conversations'][0]['exchanges'][0]
                self.assertTrue(exchange['model'].startswith('```\nusage: /{?,dir,do,file,image,url}'))
                del exchange['model']
                self.assertEqual(config, expected_config)


    def test_chat_fn_show(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            })),
            ('test.txt', 'file content')
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value

            # Create the ChatManager instance
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            chat_prompts = [f'This file:\n\n/file {temp_posix}/test.txt -n']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            mock_pool_manager_instance.request.assert_not_called()
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {
                                'user': f'''\
This file:

/file {temp_posix}/test.txt -n''',
                                'model': f'''\
This file:

<{_escape_markdown_text(temp_posix)}/test.txt>
```
file content
```
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
                            }
                        ]
                    }
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_do(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ],
                'templates': [
                    {'id': 'tmpl1', 'name': 'bye', 'title': 'Goodbye', 'prompts': ['Goodbye']}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Bye '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'bye!'}}).encode('utf-8')
            ]

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            # Create the ChatManager instance
            chat_prompts = ['/do bye']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
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
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_do_variables(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
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
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Create a mock show response
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock chat response
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Bye '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'bye!'}}).encode('utf-8')
            ]

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            # Create the ChatManager instance
            chat_prompts = ['/do bye -v name Joe']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
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
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_chat_fn_do_unknown_template(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:

            # Configure the mock session instance
            mock_pool_manager_instance = mock_pool_manager.return_value

            # Create the ChatManager instance
            chat_prompts = ['/do unknown']
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            chat_manager = ChatManager(app, 'conv1', chat_prompts)
            app.chats['conv1'] = chat_manager
            mock_thread.assert_called_once_with(target=ChatManager.chat_thread_fn, args=(chat_manager, mock_pool_manager_instance))
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            ChatManager.chat_thread_fn(chat_manager, mock_pool_manager_instance)
            mock_pool_manager_instance.request.assert_not_called()
            self.assertDictEqual(app.chats, {})

            # Verify the app config
            expected_config = {
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
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


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
        test_files = [
            ('test.txt', 'Test 1'),
            (('subdir', 'test2.txt'), 'Test 2'),
            (('subdir', 'test3.md'), '# Test 3')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'/dir {temp_posix} .txt', flags),
                f'''\
<{_escape_markdown_text(temp_posix)}/test.txt>
Test 1
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_dir_depth(self):
        test_files = [
            ('test.txt', 'Test 1'),
            (('subdir', 'test2.txt'), 'Test 2'),
            (('subdir', 'test3.md'), '# Test 3')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'/dir {temp_posix} .txt -d 2', flags),
                f'''\
<{_escape_markdown_text(temp_posix)}/subdir/test2.txt>
Test 2
</ {_escape_markdown_text(temp_posix)}/subdir/test2.txt>

<{_escape_markdown_text(temp_posix)}/test.txt>
Test 1
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_dir_exclude(self):
        test_files = [
            ('test.txt', 'Test 1'),
            (('subdir', 'test2.txt'), 'Test 2'),
            (('subdir2', 'test3.txt'), 'Test 3')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'/dir {temp_posix} .txt -d 2 -x test.txt -x subdir2/', flags),
                f'''\
<{_escape_markdown_text(temp_posix)}/subdir/test2.txt>
Test 2
</ {_escape_markdown_text(temp_posix)}/subdir/test2.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_dir_no_files(self):
        with create_test_files([]) as temp_dir, \
             self.assertRaises(ValueError) as cm_exc:
            temp_posix = pathlib.Path(temp_dir).as_posix()
            _process_commands(f'/dir {temp_posix} .txt', {})
        self.assertEqual(str(cm_exc.exception), f'no files found in directory "{temp_posix}"')


    def test_do(self):
        flags = {}
        self.assertEqual(_process_commands('/do template_name -v var1 val1', flags), 'Executing template "template_name"')
        self.assertDictEqual(flags, {'do': [('template_name', {'var1': 'val1'})]})


    def test_do_multiple(self):
        flags = {}
        self.assertEqual(
            _process_commands(
                '''\
/do template_name -v var1 val1

/do template_name -v var1 val2
''',
                flags
            ),
            '''\
Executing template "template_name"

Executing template "template_name"
'''
        )
        self.assertDictEqual(flags, {
            'do': [
                ('template_name', {'var1': 'val1'}),
                ('template_name', {'var1': 'val2'})
            ]
        })


    def test_file(self):
        test_files = [
            ('test.txt', 'file content')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_posix}/test.txt', flags),
                f'''\
<{_escape_markdown_text(temp_posix)}/test.txt>
file content
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_file_show(self):
        test_files = [
            ('test.txt', 'file content')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_posix}/test.txt -n', flags),
                f'''\
<{_escape_markdown_text(temp_posix)}/test.txt>
```
file content
```
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
            )
            self.assertDictEqual(flags, {'show': True})


    def test_file_show2(self):
        test_files = [
            ('test.txt', 'file content'),
            ('test2.txt', 'file content 2')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(
                    f'''\
/file {temp_posix}/test.txt

/file {temp_posix}/test2.txt -n
''',
                    flags
                ),
                f'''\
<{_escape_markdown_text(temp_posix)}/test.txt>
```
file content
```
</ {_escape_markdown_text(temp_posix)}/test.txt>

<{_escape_markdown_text(temp_posix)}/test2.txt>
```
file content 2
```
</ {_escape_markdown_text(temp_posix)}/test2.txt>
'''
            )
            self.assertDictEqual(flags, {'show': True})


    def test_image(self):
        test_files = [
            ('test.jpg', 'image data')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(_process_commands(f'/image {temp_posix}/test.jpg', flags), '')
            self.assertDictEqual(flags, {'images': [base64.b64encode(b'image data').decode('utf-8')]})


    def test_image_multiple(self):
        test_files = [
            ('test.jpg', 'image data'),
            ('test2.jpg', 'image data 2')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(
                    f'''\
/image {temp_posix}/test.jpg

/image {temp_posix}/test2.jpg
''',
                    flags
                ),
                '\n\n\n'
            )
            self.assertDictEqual(flags, {
                'images': [
                    base64.b64encode(b'image data').decode('utf-8'),
                    base64.b64encode(b'image data 2').decode('utf-8')
                ]
            })


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
        test_files = [
            ('test.txt', 'file content')
        ]
        with create_test_files(test_files) as temp_dir:
            temp_posix = str(pathlib.Path(temp_dir).as_posix())
            flags = {}
            self.assertEqual(
                _process_commands(f'Hello\n\n/?\n\n/file {temp_posix}/test.txt', flags),
                f'''\
Hello

Displaying top-level help

<{_escape_markdown_text(temp_posix)}/test.txt>
file content
</ {_escape_markdown_text(temp_posix)}/test.txt>'''
            )
            self.assertIn('help', flags)


    def test_file_error(self):
        with create_test_files([]) as temp_dir:
            flags = {}
            with self.assertRaises(FileNotFoundError):
                _process_commands(f'/file {temp_dir}/nonexistent.txt', flags)
