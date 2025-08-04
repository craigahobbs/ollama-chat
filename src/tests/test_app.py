# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import json
from io import StringIO
import os
import re
import unittest
import unittest.mock

import urllib3
from schema_markdown import encode_query_string
from ollama_chat.app import DownloadManager, OllamaChat

from .util import create_test_files


class TestApp(unittest.TestCase):

    def test_init(self):
        test_files = [
            ('ollama-chat.json', json.dumps({'model': 'llm', 'conversations': []}))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            self.assertEqual(app.config.config_path, config_path)
            with app.config() as config:
                self.assertDictEqual(config, {'model': 'llm', 'conversations': []})
            self.assertDictEqual(app.chats, {})
            self.assertListEqual(
                sorted(request.name for request in app.requests.values() if request.doc_group.startswith('Ollama Chat ')),
                [
                    'createTemplate',
                    'deleteConversation',
                    'deleteConversationExchange',
                    'deleteModel',
                    'deleteTemplate',
                    'downloadModel',
                    'getConversation',
                    'getConversations',
                    'getModels',
                    'getSystemInfo',
                    'getTemplate',
                    'index.html',
                    'moveConversation',
                    'moveTemplate',
                    'ollamaChat.bare',
                    'ollamaChatConversation.bare',
                    'ollamaChatModels.bare',
                    'ollamaChatTemplate.bare',
                    'regenerateConversationExchange',
                    'replyConversation',
                    'setConversationTitle',
                    'setModel',
                    'startConversation',
                    'startTemplate',
                    'stopConversation',
                    'stopModelDownload',
                    'updateTemplate'
                ]
            )


    def test_init_missing_config(self):
        with unittest.mock.patch('os.path.isfile', return_value=False) as mock_isfile:
            app = OllamaChat('ollama-chat.json')
            mock_isfile.assert_called_once_with('ollama-chat.json')
            self.assertEqual(app.config.config_path, 'ollama-chat.json')
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})
            self.assertDictEqual(app.chats, {})
            self.assertListEqual(
                sorted(request.name for request in app.requests.values() if request.doc_group.startswith('Ollama Chat ')),
                [
                    'createTemplate',
                    'deleteConversation',
                    'deleteConversationExchange',
                    'deleteModel',
                    'deleteTemplate',
                    'downloadModel',
                    'getConversation',
                    'getConversations',
                    'getModels',
                    'getSystemInfo',
                    'getTemplate',
                    'index.html',
                    'moveConversation',
                    'moveTemplate',
                    'ollamaChat.bare',
                    'ollamaChatConversation.bare',
                    'ollamaChatModels.bare',
                    'ollamaChatTemplate.bare',
                    'regenerateConversationExchange',
                    'replyConversation',
                    'setConversationTitle',
                    'setModel',
                    'startConversation',
                    'startTemplate',
                    'stopConversation',
                    'stopModelDownload',
                    'updateTemplate'
                ]
            )


class TestDownloadManager(unittest.TestCase):

    def test_download_fn(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.read_chunked.return_value = [
                json.dumps({'status': 'success', 'completed': 1000, 'total': 2000}).encode('utf-8'),
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            # Create the ChatManager instance
            download_manager = DownloadManager(app, 'llm:7b')
            app.downloads['llm:7b'] = download_manager
            mock_thread.assert_called_once_with(
                target=DownloadManager.download_thread_fn,
                args=(download_manager, mock_pool_manager_instance)
            )
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            DownloadManager.download_thread_fn(download_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.downloads, {})
            self.assertEqual(download_manager.status, 'success')
            self.assertEqual(download_manager.completed, 1000)
            self.assertEqual(download_manager.total, 2000)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_download_fn_stop(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.read_chunked.return_value = [
                json.dumps({'status': 'success', 'completed': 1000, 'total': 2000}).encode('utf-8'),
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            # Create the ChatManager instance
            download_manager = DownloadManager(app, 'llm:7b')
            download_manager.stop = True
            mock_thread.assert_called_once_with(
                target=DownloadManager.download_thread_fn,
                args=(download_manager, mock_pool_manager_instance)
            )
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            DownloadManager.download_thread_fn(download_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.downloads, {})
            self.assertEqual(download_manager.status, '')
            self.assertEqual(download_manager.completed, 0)
            self.assertEqual(download_manager.total, 0)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_download_fn_ollama_failure(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('threading.Thread') as mock_thread, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            # Create the ChatManager instance
            download_manager = DownloadManager(app, 'llm:7b')
            app.downloads['llm:7b'] = download_manager
            mock_thread.assert_called_once_with(
                target=DownloadManager.download_thread_fn,
                args=(download_manager, mock_pool_manager_instance)
            )
            mock_thread.return_value.start.assert_called_once_with()
            self.assertTrue(mock_thread.return_value.daemon)

            # Run the thread function
            DownloadManager.download_thread_fn(download_manager, mock_pool_manager_instance)
            self.assertDictEqual(app.downloads, {})
            self.assertEqual(download_manager.status, '')
            self.assertEqual(download_manager.completed, 0)
            self.assertEqual(download_manager.total, 0)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


class TestAPI(unittest.TestCase):

    def test_xorigin(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path, xorigin=True)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json'), ('Access-Control-Allow-Origin', '*')])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_get_conversations(self):
        test_files = [
            ('ollama-chat.json', json.dumps({'model': 'llm', 'conversations': []}))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})

            # Verify the app config
            expected_config = {'model': 'llm', 'conversations': []}
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertDictEqual(json.load(config_fh), expected_config)


    def test_get_conversations_no_model(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_set_model(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            status, headers, content_bytes = app.request('POST', '/setModel', wsgi_input=json.dumps({'model': 'llm'}).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {})

            # Verify the app config
            expected_config = {'model': 'llm', 'conversations': []}
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_move_conversation_down(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv2 down
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_move_conversation_up(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv2 up
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_move_conversation_down_last(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv3 down
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv3', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_move_conversation_up_first(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv1 up
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv1', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_move_conversation_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to move a non-existent conversation
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_move_template_down(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl2 down
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_move_template_up(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl2 up
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_move_template_down_last(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl3 down
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl3', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_move_template_up_first(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl1 up
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl1', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_move_template_unknown_id(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to move tmpl2
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)



    def test_delete_template_success(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete template 'tmpl1'
            status, headers, content_bytes = app.request(
                'POST', '/deleteTemplate', wsgi_input=json.dumps({'id': 'tmpl1'}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_delete_template_unknown_id(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete 'tmpl2'
            status, headers, content_bytes = app.request(
                'POST', '/deleteTemplate', wsgi_input=json.dumps({'id': 'tmpl2'}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_template_no_templates(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete when templates key doesn't exist
            status, headers, content_bytes = app.request(
                'POST', '/deleteTemplate', wsgi_input=json.dumps({'id': 'tmpl1'}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_template_success(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']},
                {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Get template 'tmpl1'
            status, headers, content_bytes = app.request(
                'GET', '/getTemplate', query_string=encode_query_string({'id': 'tmpl1'})
            )
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(
                response,
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            )

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_template_unknown_id(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to get non-existent template 'tmpl2'
            status, headers, content_bytes = app.request(
                'GET', '/getTemplate', query_string=encode_query_string({'id': 'tmpl2'})
            )
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_template_no_templates(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to get template when no templates exist
            status, headers, content_bytes = app.request(
                'GET', '/getTemplate', query_string=encode_query_string({'id': 'tmpl1'})
            )
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_update_template_success(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']},
                {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Update template 'tmpl1'
            updated_template = {
                'id': 'tmpl1',
                'title': 'Updated Template 1',
                'prompts': ['Updated Prompt'],
                'variables': [{'name': 'var1', 'label': 'Variable 1'}]
            }
            status, headers, content_bytes = app.request(
                'POST', '/updateTemplate',
                wsgi_input=json.dumps(updated_template).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    updated_template,
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_update_template_unknown_id(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update non-existent template 'tmpl2'
            updated_template = {'id': 'tmpl2', 'title': 'Updated Template 2', 'prompts': ['New Prompt']}
            status, headers, content_bytes = app.request(
                'POST', '/updateTemplate',
                wsgi_input=json.dumps(updated_template).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_update_template_no_templates(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            updated_template = {'id': 'tmpl1', 'title': 'New Template', 'prompts': ['New Prompt']}
            status, headers, content_bytes = app.request(
                'POST', '/updateTemplate',
                wsgi_input=json.dumps(updated_template).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_conversation(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'user': 'Hello'}
            status, headers, content_bytes = app.request('POST', '/startConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], ['Hello'])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                'model': 'llm',
                'conversations': [
                    {'id': '12345678-1234-5678-1234-567812345678', 'model': 'llm', 'title': 'Hello', 'exchanges': []}
                ]
            })

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_conversation_model(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'model': 'llm', 'user': 'Hello'}
            status, headers, content_bytes = app.request('POST', '/startConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], ['Hello'])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'model': 'llm', 'title': 'Hello', 'exchanges': []}
                    ]
                })

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_start_conversation_max_title(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            prompt = 'This is a very long title and needs to be cut off because its too long'
            request = {'model': 'llm', 'user': prompt}
            status, headers, content_bytes = app.request('POST', '/startConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], [prompt])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'This is a very long title and needs to be cut o...',
                            'exchanges': []
                        }
                    ]
                })

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_start_conversation_no_model(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'user': 'Hello'}
            status, headers, content_bytes = app.request('POST', '/startConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'NoModel'})
            mock_manager.assert_not_called()
            self.assertNotIn('12345678-1234-5678-1234-567812345678', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_start_template(self):
        original_config = {
            'model': 'llm',
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'tmpl1'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], ['Prompt 1'])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'model': 'llm',
                    'conversations': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'model': 'llm', 'title': 'Template 1', 'exchanges': []}
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_template_by_name(self):
        original_config = {
            'model': 'llm',
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'test'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], ['Prompt 1'])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'model': 'llm',
                    'conversations': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'model': 'llm', 'title': 'Template 1', 'exchanges': []}
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_template_model(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value = '12345678-1234-5678-1234-567812345678'), \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'model': 'llm', 'id': 'tmpl1'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})
            mock_manager.assert_called_once_with(app, response['id'], ['Prompt 1'])
            self.assertIs(app.chats['12345678-1234-5678-1234-567812345678'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'model': 'llm', 'title': 'Template 1', 'exchanges': []}
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_template_no_model(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'tmpl1'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'NoModel'})
            mock_manager.assert_not_called()
            self.assertNotIn('12345678-1234-5678-1234-567812345678', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_template_unknown_template(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'unknown'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID', 'message': 'Unknown template "unknown"'})
            mock_manager.assert_not_called()
            self.assertNotIn('12345678-1234-5678-1234-567812345678', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_start_template_unknown_variable(self):
        original_config = {
            'conversations': [],
            'templates': [
                {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'tmpl1', 'variables': {'foo': 'bar'}}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownVariable', 'message': 'unknown variable "foo"'})
            mock_manager.assert_not_called()
            self.assertNotIn('12345678-1234-5678-1234-567812345678', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_start_template_missing_variable(self):
        original_config = {
            'conversations': [],
            'templates': [
                {
                    'id': 'tmpl1',
                    'name': 'test',
                    'title': 'Template 1',
                    'variables': [{'name': 'name', 'label': 'Name'}],
                    'prompts': ['Prompt {{name}}'],
                }
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            request = {'id': 'tmpl1'}
            status, headers, content_bytes = app.request('POST', '/startTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'MissingVariable', 'message': 'missing variable value for "name"'})
            mock_manager.assert_not_called()
            self.assertNotIn('12345678-1234-5678-1234-567812345678', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_stop_conversation_success(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock ChatManager instance
            mock_chat = unittest.mock.Mock()
            mock_chat.stop = False
            app.chats['conv1'] = mock_chat

            # Stop the conversation
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request('POST', '/stopConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertTrue(mock_chat.stop)
            self.assertNotIn('conv1', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_stop_conversation_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to stop non-existent conversation
            request = {'id': 'conv2'}
            status, headers, content_bytes = app.request('POST', '/stopConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})
            self.assertNotIn('conv2', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_stop_conversation_not_generating(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to stop conversation that's not generating
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request('POST', '/stopConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertDictEqual(app.chats, {})
            self.assertNotIn('conv1', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_conversation_success_not_generating(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Get conversation 'conv1'
            status, headers, content_bytes = app.request('GET', '/getConversation', query_string=encode_query_string({'id': 'conv1'}))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(
                response,
                {
                    'conversation': {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}],
                        'generating': False
                    }
                }
            )

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_conversation_success_generating(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Add a mock ChatManager to simulate generating state
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Get conversation 'conv1'
            status, headers, content_bytes = app.request('GET', '/getConversation', query_string=encode_query_string({'id': 'conv1'}))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(
                response,
                {
                    'conversation': {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}],
                        'generating': True
                    }
                }
            )

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_conversation_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to get non-existent conversation 'conv2'
            status, headers, content_bytes = app.request('GET', '/getConversation', query_string=encode_query_string({'id': 'conv2'}))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_reply_conversation_success(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Reply to conversation 'conv1'
            request = {'id': 'conv1', 'user': 'How are you?'}
            status, headers, content_bytes = app.request('POST', '/replyConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            mock_manager.assert_called_once_with(app, 'conv1', ['How are you?'])
            self.assertIs(app.chats['conv1'], mock_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_reply_conversation_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to reply to non-existent conversation 'conv2'
            request = {'id': 'conv2', 'user': 'How are you?'}
            status, headers, content_bytes = app.request('POST', '/replyConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})
            mock_manager.assert_not_called()
            self.assertNotIn('conv2', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_reply_conversation_busy(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Simulate a busy conversation by adding it to chats
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Try to reply to busy conversation 'conv1'
            request = {'id': 'conv1', 'user': 'How are you?'}
            status, headers, content_bytes = app.request('POST', '/replyConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'ConversationBusy'})
            mock_manager.assert_not_called()
            self.assertEqual(app.chats['conv1'], mock_chat)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_set_conversation_title_success(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Set new title for conversation 'conv1'
            request = {'id': 'conv1', 'title': 'New Title'}
            status, headers, content_bytes = app.request('POST', '/setConversationTitle', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'New Title', 'exchanges': []}]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_set_conversation_title_unknown_id(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to set title for non-existent conversation 'conv2'
            request = {'id': 'conv2', 'title': 'New Title'}
            status, headers, content_bytes = app.request('POST', '/setConversationTitle', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_set_conversation_title_busy(self):
        original_config = {
            'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Simulate a busy conversation by adding it to chats
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Try to set title for busy conversation 'conv1'
            request = {'id': 'conv1', 'title': 'New Title'}
            status, headers, content_bytes = app.request('POST', '/setConversationTitle', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'ConversationBusy'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_conversation_success(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete conversation 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request('POST', '/deleteConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_delete_conversation_unknown_id(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete non-existent conversation 'conv2'
            request = {'id': 'conv2'}
            status, headers, content_bytes = app.request('POST', '/deleteConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_delete_conversation_busy(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Simulate a busy conversation by adding it to chats
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Try to delete busy conversation 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request('POST', '/deleteConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'ConversationBusy'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_create_template_basic(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('uuid.uuid4', return_value='12345678-1234-5678-1234-567812345678'):
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a basic template
            request = {'title': 'New Template', 'prompts': ['New Prompt']}
            status, headers, content_bytes = app.request('POST', '/createTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {'id': '12345678-1234-5678-1234-567812345678', 'title': 'New Template', 'prompts': ['New Prompt']},
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_create_template_no_templates(self):
        with create_test_files([]) as temp_dir, \
        unittest.mock.patch('uuid.uuid4', return_value='12345678-1234-5678-1234-567812345678'):
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a basic template
            request = {'title': 'New Template', 'prompts': ['New Prompt']}
            status, headers, content_bytes = app.request('POST', '/createTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {'id': '12345678-1234-5678-1234-567812345678', 'title': 'New Template', 'prompts': ['New Prompt']}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_create_template_optional_arguments(self):
        with create_test_files([]) as temp_dir, \
        unittest.mock.patch('uuid.uuid4', return_value='12345678-1234-5678-1234-567812345678'):
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a template with all fields
            request = {
                'title': 'New Template',
                'prompts': ['New Prompt'],
                'name': 'test_template',
                'variables': [{'name': 'var1', 'label': 'Variable 1'}]
            }
            status, headers, content_bytes = app.request('POST', '/createTemplate', wsgi_input=json.dumps(request).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'id': '12345678-1234-5678-1234-567812345678'})

            # Verify the app config
            expected_config = {
                'conversations': [],
                'templates': [
                    {
                        'id': '12345678-1234-5678-1234-567812345678',
                        'title': 'New Template',
                        'prompts': ['New Prompt'],
                        'name': 'test_template',
                        'variables': [{'name': 'var1', 'label': 'Variable 1'}]
                    }
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_delete_conversation_exchange_success(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {'user': 'Hello', 'model': 'Hi there'},
                            {'user': 'How are you?', 'model': 'Good'}
                        ]
                    }
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete the most recent exchange from 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/deleteConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_delete_conversation_exchange_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete exchange from non-existent conversation 'conv2'
            request = {'id': 'conv2'}
            status, headers, content_bytes = app.request(
                'POST', '/deleteConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_conversation_exchange_busy(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Simulate a busy conversation
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Try to delete exchange from busy conversation 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/deleteConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'ConversationBusy'})
            self.assertIs(app.chats['conv1'], mock_chat)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_conversation_exchange_empty(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete exchange from conversation with no exchanges
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/deleteConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertNotIn('conv1', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_regenerate_conversation_exchange_success(self):
        test_files = [
            ('ollama-chat.json', json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [
                            {'user': 'Hello', 'model': 'Hi there'},
                            {'user': 'How are you?', 'model': 'Good'}
                        ]
                    }
                ]
            }))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Regenerate the last exchange for 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/regenerateConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check that ChatManager was called with the last user prompt
            mock_manager.assert_called_once_with(app, 'conv1', ['How are you?'])
            self.assertIs(app.chats['conv1'], mock_manager.return_value)

            # Verify the app config
            expected_config = {
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
                ]
            }
            with app.config() as config:
                self.assertDictEqual(config, expected_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), expected_config)


    def test_regenerate_conversation_exchange_unknown_id(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to regenerate exchange for non-existent conversation 'conv2'
            request = {'id': 'conv2'}
            status, headers, content_bytes = app.request(
                'POST', '/regenerateConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})
            mock_manager.assert_not_called()
            self.assertNotIn('conv2', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_regenerate_conversation_exchange_busy(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Simulate a busy conversation
            mock_chat = unittest.mock.Mock()
            app.chats['conv1'] = mock_chat

            # Try to regenerate exchange for busy conversation 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/regenerateConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'ConversationBusy'})
            mock_manager.assert_not_called()
            self.assertIs(app.chats['conv1'], mock_chat)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_regenerate_conversation_exchange_empty(self):
        original_config = {
            'conversations': [
                {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
            ]
        }
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.ChatManager') as mock_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to regenerate exchange for conversation with no exchanges
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request(
                'POST', '/regenerateConversationExchange', wsgi_input=json.dumps(request).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            mock_manager.assert_not_called()
            self.assertNotIn('conv1', app.chats)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_success(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
                ('ollama-chat.json', json.dumps(original_config))
             ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 200
            mock_list_response.json.return_value = {
                'models': [
                    {
                        'model': 'llm:7b',
                        'details': {'parameter_size': '7B'},
                        'size': 4100000000,
                        'modified_at': '2023-10-01T12:00:00+00:00'
                    },
                    {
                        'model': 'other:tag',
                        'details': {'parameter_size': '3M'},
                        'size': 1800000,
                        'modified_at': '2023-10-02T12:00:00+00:00'
                    },
                    {
                        'model': 'other2:tag',
                        'details': {'parameter_size': '3K'},
                        'size': 1800,
                        'modified_at': '2023-10-02T12:00:00+00:00'
                    }
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [
                    {'id': 'llm:7b', 'name': 'llm', 'parameters': 7000000000, 'size': 4100000000, 'modified': '2023-10-01T12:00:00+00:00'},
                    {'id': 'other2:tag', 'modified': '2023-10-02T12:00:00+00:00', 'name': 'other2', 'parameters': 3000, 'size': 1800},
                    {'id': 'other:tag', 'name': 'other', 'parameters': 3000000, 'size': 1800000, 'modified': '2023-10-02T12:00:00+00:00'}
                ],
                'downloading': [],
                'model': 'llm'
            })

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_parameter_size_warning(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
                ('ollama-chat.json', json.dumps(original_config))
             ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 200
            mock_list_response.json.return_value = {
                'models': [
                    {
                        'model': 'llm:7b',
                        'details': {'parameter_size': '1000'},
                        'size': 1000,
                        'modified_at': '2023-10-01T12:00:00+00:00'
                    }
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            environ = {'wsgi.errors': StringIO()}
            status, headers, content_bytes = app.request('GET', '/getModels', environ=environ)
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [
                    {'id': 'llm:7b', 'name': 'llm', 'parameters': 0, 'size': 1000, 'modified': '2023-10-01T12:00:00+00:00'}
                ],
                'downloading': [],
                'model': 'llm'
            })
            logs = environ['wsgi.errors'].getvalue()
            logs = re.sub(r'\[.*?\]', '[X / Y]', logs)
            self.assertEqual(logs, 'WARNING [X / Y] Invalid parameter size "1000"\n')

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_no_models(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 200
            mock_list_response.json.return_value = {
                'models': []
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'models': [], 'downloading': [], 'model': 'llm'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_downloading(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 200
            mock_list_response.json.return_value = {'models': []}

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            # Add downloading models
            mock_download = unittest.mock.Mock()
            mock_download.status = 'downloading'
            mock_download.completed = 5000000
            mock_download.total = 10000000
            app.downloads['downloading_model'] = mock_download
            mock_download2 = unittest.mock.Mock()
            mock_download2.status = 'unknown'
            mock_download2.completed = 0
            mock_download2.total = 0
            app.downloads['downloading_model2'] = mock_download2

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [],
                'downloading': [
                    {'id': 'downloading_model', 'status': 'downloading', 'completed': 5000000, 'size': 10000000},
                    {'id': 'downloading_model2', 'status': 'unknown', 'completed': 0}
                ],
                'model': 'llm'
            })

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_ollama_failure(self):
        original_config = {'model': 'llm', 'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '500 Internal Server Error')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'error': 'UnexpectedError'})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_models_no_config_model(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the list request
            mock_list_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_list_response.status = 200
            mock_list_response.json.return_value = {
                'models': [
                    {
                        'model': 'llm:7b',
                        'details': {'parameter_size': '7B'},
                        'size': 4100000000,
                        'modified_at': '2023-10-01T12:00:00+00:00'
                    }
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_list_response

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [
                    {'id': 'llm:7b', 'name': 'llm', 'parameters': 7000000000, 'size': 4100000000, 'modified': '2023-10-01T12:00:00+00:00'}
                ],
                'downloading': []
            })

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))


    def test_download_model_success(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('ollama_chat.app.DownloadManager') as mock_download_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Initiate model download
            request = {'model': 'llm:7b'}
            status, headers, content_bytes = app.request('POST', '/downloadModel', wsgi_input=json.dumps(request).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {})

            # Verify DownloadManager was called and stored
            mock_download_manager.assert_called_once_with(app, 'llm:7b')
            self.assertIn('llm:7b', app.downloads)
            self.assertIs(app.downloads['llm:7b'], mock_download_manager.return_value)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_stop_model_download_success(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Add a mock download to the downloads dictionary
            mock_download = unittest.mock.Mock()
            mock_download.stop = False
            app.downloads['llm:7b'] = mock_download

            # Stop the model download
            request = {'model': 'llm:7b'}
            status, headers, content_bytes = app.request('POST', '/stopModelDownload', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertTrue(mock_download.stop)  # Verify stop flag was set
            self.assertIs(app.downloads['llm:7b'], mock_download)

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_stop_model_download_not_downloading(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to stop a non-existent download
            request = {'model': 'llm:7b'}
            status, headers, content_bytes = app.request('POST', '/stopModelDownload', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertDictEqual(app.downloads, {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_model_success(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the delete request
            mock_delete_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_delete_response.status = 200

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_delete_response

            # Delete model 'llm:7b'
            request = {'model': 'llm:7b'}
            status, headers, content_bytes = app.request('POST', '/deleteModel', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            mock_pool_manager_instance.request.assert_called_once_with(
                'DELETE', 'http://127.0.0.1:11434/api/delete', json={'model': 'llm:7b'}, retries=unittest.mock.ANY
            )

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_delete_model_error(self):
        original_config = {'conversations': []}
        test_files = [
            ('ollama-chat.json', json.dumps(original_config))
        ]
        with create_test_files(test_files) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create a mock Response object for the delete request
            mock_delete_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_delete_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_delete_response

            # Delete model 'llm:7b'
            request = {'model': 'llm:7b'}
            status, headers, content_bytes = app.request('POST', '/deleteModel', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '500 Internal Server Error')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {'error': 'UnexpectedError'})
            mock_pool_manager_instance.request.assert_called_once_with(
                'DELETE', 'http://127.0.0.1:11434/api/delete', json={'model': 'llm:7b'}, retries=unittest.mock.ANY
            )

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, original_config)

            # Verify the config file
            with open(config_path, 'r', encoding='utf-8') as config_fh:
                self.assertEqual(json.load(config_fh), original_config)


    def test_get_system_info(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Make the request
            status, headers, content_bytes = app.request('GET', '/getSystemInfo')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertTrue(response['memory'] > 0)
            del response['memory']
            self.assertDictEqual(response, {})

            # Verify the app config
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            # Verify the config file
            self.assertFalse(os.path.exists(config_path))
