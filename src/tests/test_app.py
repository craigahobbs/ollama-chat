# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import json
import os
import unittest
import unittest.mock

from ollama_chat.app import OllamaChat

from .util import create_test_files


class TestApp(unittest.TestCase):

    def test_init(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
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


    def test_get_conversations(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})


    def test_get_conversations_no_model(self):
        with create_test_files([]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})


    def test_set_model(self):
        with create_test_files([]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)
            with app.config() as config:
                self.assertDictEqual(config, {'conversations': []})

            status, headers, content_bytes = app.request('POST', '/setModel', wsgi_input=json.dumps({'model': 'llm'}).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {})

            with app.config() as config:
                self.assertDictEqual(config, {'model': 'llm', 'conversations': []})


    def test_move_conversation_down(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv2 down
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                        {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []},
                        {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                    ]
                })


    def test_move_conversation_up(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv2 up
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                        {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                        {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                    ]
                })


    def test_move_conversation_down_last(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv3 down
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv3', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                        {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                        {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                    ]
                })


    def test_move_conversation_up_first(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                    {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move conv1 up
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv1', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                        {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []},
                        {'id': 'conv3', 'model': 'llm', 'title': 'Conversation 3', 'exchanges': []}
                    ]
                })


    def test_move_conversation_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to move a non-existent conversation
            status, headers, content_bytes = app.request(
                'POST', '/moveConversation', wsgi_input=json.dumps({'id': 'conv2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})


    def test_move_template_down(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl2 down
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                        {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []},
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                    ]
                })


    def test_move_template_up(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl2 up
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                        {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                    ]
                })


    def test_move_template_down_last(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl3 down
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl3', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                        {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                    ]
                })


    def test_move_template_up_first(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                    {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Move tmpl1 up
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl1', 'down': False}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []},
                        {'id': 'tmpl3', 'title': 'Template 3', 'prompts': []}
                    ]
                })


    def test_move_template_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to move tmpl2
            status, headers, content_bytes = app.request(
                'POST', '/moveTemplate', wsgi_input=json.dumps({'id': 'tmpl2', 'down': True}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})



    def test_delete_template_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete template 'tmpl1'
            status, headers, content_bytes = app.request(
                'POST', '/deleteTemplate', wsgi_input=json.dumps({'id': 'tmpl1'}).encode('utf-8')
            )
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                    ]
                })


    def test_delete_template_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []}
                ]
            }))
        ]) as input_dir:
            config_path = os.path.join(input_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to delete 'tmpl2'
            status, headers, content_bytes = app.request(
                'POST', '/deleteTemplate', wsgi_input=json.dumps({'id': 'tmpl2'}).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})
