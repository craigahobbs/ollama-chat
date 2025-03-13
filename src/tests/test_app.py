# {% raw %}
# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import ctypes
import datetime
import json
import os
import platform
import unittest
import unittest.mock

from schema_markdown import encode_query_string
from ollama_chat.app import OllamaChat, MEMORYSTATUSEX

from .util import create_test_files


class TestApp(unittest.TestCase):

    def test_init(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir:
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


class TestAPI(unittest.TestCase):

    def test_get_conversations(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'model': 'llm', 'conversations': [], 'templates': []})


    def test_get_conversations_no_model(self):
        with create_test_files([]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            status, headers, content_bytes = app.request('GET', '/getConversations')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'conversations': [], 'templates': []})


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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
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



    def test_delete_template_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': []},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': []}
                ]
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
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
        ]) as temp_dir:
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


    def test_delete_template_no_templates(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': []
            }))
        ]) as temp_dir:
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': []
                })


    def test_get_template_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
                ]
            }))
        ]) as temp_dir:
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


    def test_get_template_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir:
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


    def test_get_template_no_templates(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': []
            }))
        ]) as temp_dir:
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


    def test_update_template_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']},
                    {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
                ]
            }))
        ]) as temp_dir:
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

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        updated_template,
                        {'id': 'tmpl2', 'title': 'Template 2', 'prompts': ['Prompt 2']}
                    ]
                })


    def test_update_template_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update non-existent template 'tmpl2'
            updated_template = {
                'id': 'tmpl2',
                'title': 'Updated Template 2',
                'prompts': ['New Prompt']
            }
            status, headers, content_bytes = app.request(
                'POST', '/updateTemplate',
                wsgi_input=json.dumps(updated_template).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


    def test_update_template_no_templates(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': []
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to update template when no templates exist
            updated_template = {
                'id': 'tmpl1',
                'title': 'New Template',
                'prompts': ['New Prompt']
            }
            status, headers, content_bytes = app.request(
                'POST', '/updateTemplate',
                wsgi_input=json.dumps(updated_template).encode('utf-8')
            )
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertDictEqual(response, {'error': 'UnknownTemplateID'})

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': []
                })


    def test_start_conversation(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'model': 'llm',
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'Hello',
                            'exchanges': []
                        }
                    ]
                })


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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'Hello',
                            'exchanges': []
                        }
                    ]
                })


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

            # Check config
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': []
                })


    def test_start_template(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'model': 'llm',
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'model': 'llm',
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'Template 1',
                            'exchanges': []
                        }
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


    def test_start_template_by_name(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'model': 'llm',
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'model': 'llm',
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'Template 1',
                            'exchanges': []
                        }
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


    def test_start_template_model(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': '12345678-1234-5678-1234-567812345678',
                            'model': 'llm',
                            'title': 'Template 1',
                            'exchanges': []
                        }
                    ],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


    def test_start_template_no_model(self):
        with create_test_files([
                (('ollama-chat.json',), json.dumps({
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                }))
             ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': []
                })


    def test_start_template_unknown_variable(self):
        with create_test_files([
                (('ollama-chat.json',), json.dumps({
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                }))
             ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': 'tmpl1', 'name': 'test', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


    def test_start_template_missing_variable(self):
        with create_test_files([
                (('ollama-chat.json',), json.dumps({
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
                }))
             ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
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
                })


    def test_stop_conversation_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir:
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


    def test_stop_conversation_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir:
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


    def test_stop_conversation_not_generating(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir:
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


    def test_get_conversation_success_not_generating(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir:
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


    def test_get_conversation_success_generating(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir:
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


    def test_get_conversation_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to get non-existent conversation 'conv2'
            status, headers, content_bytes = app.request('GET', '/getConversation', query_string=encode_query_string({'id': 'conv2'}))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '400 Bad Request')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'error': 'UnknownConversationID'})


    def test_reply_conversation_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir, \
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


    def test_reply_conversation_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir, \
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


    def test_reply_conversation_busy(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir, \
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


    def test_set_conversation_title_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}]
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Set new title for conversation 'conv1'
            request = {'id': 'conv1', 'title': 'New Title'}
            status, headers, content_bytes = app.request('POST', '/setConversationTitle', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'New Title', 'exchanges': []}]
                })


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


    def test_set_conversation_title_busy(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}]
            }))
        ]) as temp_dir:
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [{'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}]
                })
            self.assertIs(app.chats['conv1'], mock_chat)


    def test_delete_conversation_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []},
                    {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                ]
            }))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete conversation 'conv1'
            request = {'id': 'conv1'}
            status, headers, content_bytes = app.request('POST', '/deleteConversation', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv2', 'model': 'llm', 'title': 'Conversation 2', 'exchanges': []}
                    ]
                })
            self.assertNotIn('conv1', app.chats)


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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': []
                })
            self.assertNotIn('conv2', app.chats)


    def test_delete_conversation_busy(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                ]
            }))
        ]) as temp_dir:
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {'id': 'conv1', 'model': 'llm', 'title': 'Conversation 1', 'exchanges': []}
                    ]
                })
            self.assertIs(app.chats['conv1'], mock_chat)


    def test_create_template_basic(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [],
                'templates': [
                    {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                ]
            }))
        ]) as temp_dir, \
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

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'title': 'New Template', 'prompts': ['New Prompt']},
                        {'id': 'tmpl1', 'title': 'Template 1', 'prompts': ['Prompt 1']}
                    ]
                })


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

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [],
                    'templates': [
                        {'id': '12345678-1234-5678-1234-567812345678', 'title': 'New Template', 'prompts': ['New Prompt']}
                    ]
                })


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

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
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
                })


    def test_delete_conversation_exchange_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
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
        ]) as temp_dir:
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

            # Check the config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {'user': 'Hello', 'model': 'Hi there'}
                            ]
                        }
                    ]
                })
            self.assertNotIn('conv1', app.chats)


    def test_delete_conversation_exchange_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir:
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


    def test_delete_conversation_exchange_busy(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir:
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


    def test_delete_conversation_exchange_empty(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': []
                    }
                ]
            }))
        ]) as temp_dir:
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': []
                        }
                    ]
                })
            self.assertNotIn('conv1', app.chats)


    def test_regenerate_conversation_exchange_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
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
        ]) as temp_dir, \
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

            # Check the config - last exchange should be removed
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [
                                {'user': 'Hello', 'model': 'Hi there'}
                            ]
                        }
                    ]
                })

            # Check that ChatManager was called with the last user prompt
            mock_manager.assert_called_once_with(app, 'conv1', ['How are you?'])
            self.assertIs(app.chats['conv1'], mock_manager.return_value)


    def test_regenerate_conversation_exchange_unknown_id(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir, \
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


    def test_regenerate_conversation_exchange_busy(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                    }
                ]
            }))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': [{'user': 'Hello', 'model': 'Hi there'}]
                        }
                    ]
                })


    def test_regenerate_conversation_exchange_empty(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({
                'conversations': [
                    {
                        'id': 'conv1',
                        'model': 'llm',
                        'title': 'Conversation 1',
                        'exchanges': []
                    }
                ]
            }))
        ]) as temp_dir, \
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

            # Check config
            with app.config() as config:
                self.assertDictEqual(config, {
                    'conversations': [
                        {
                            'id': 'conv1',
                            'model': 'llm',
                            'title': 'Conversation 1',
                            'exchanges': []
                        }
                    ]
                })


    def test_get_models_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama.list') as mock_ollama_list:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create mock models
            model1 = unittest.mock.Mock()
            model1.model = 'llm:latest'
            model1.details = unittest.mock.Mock(parameter_size='7B')
            model1.size = 4100000000
            model1.modified_at = datetime.datetime.fromisoformat('2023-10-01T12:00:00+00:00')
            model2 = unittest.mock.Mock()
            model2.model = 'other:tag'
            model2.details = unittest.mock.Mock(parameter_size='3M')
            model2.size = 1800000
            model2.modified_at = datetime.datetime.fromisoformat('2023-10-02T12:00:00+00:00')
            model3 = unittest.mock.Mock()
            model3.model = 'other2:tag'
            model3.details = unittest.mock.Mock(parameter_size='3K')
            model3.size = 1800
            model3.modified_at = datetime.datetime.fromisoformat('2023-10-02T12:00:00+00:00')
            mock_ollama_list.return_value = {'models': [model1, model2, model3]}

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [
                    {
                        'id': 'llm:latest',
                        'name': 'llm',
                        'parameters': 7000000000,
                        'size': 4100000000,
                        'modified': '2023-10-01T12:00:00+00:00'
                    },
                    {
                        'id': 'other2:tag',
                        'modified': '2023-10-02T12:00:00+00:00',
                        'name': 'other2',
                        'parameters': 3000,
                        'size': 1800
                    },
                    {
                        'id': 'other:tag',
                        'name': 'other',
                        'parameters': 3000000,
                        'size': 1800000,
                        'modified': '2023-10-02T12:00:00+00:00'
                    }
                ],
                'downloading': [],
                'model': 'llm'
            })


    def test_get_models_no_models(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama.list') as mock_ollama_list:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Mock ollama.list to return no models
            mock_ollama_list.return_value = {'models': []}

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [],
                'downloading': [],
                'model': 'llm'
            })


    def test_get_models_downloading(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama.list') as mock_ollama_list:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Mock ollama.list to return no models
            mock_ollama_list.return_value = {'models': []}

            # Add a downloading model
            mock_download = unittest.mock.Mock()
            mock_download.status = 'downloading'
            mock_download.completed = 5000000
            mock_download.total = 10000000
            app.downloads['downloading_model'] = mock_download

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [],
                'downloading': [
                    {
                        'id': 'downloading_model',
                        'status': 'downloading',
                        'completed': 5000000,
                        'size': 10000000
                    }
                ],
                'model': 'llm'
            })


    def test_get_models_ollama_failure(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'model': 'llm', 'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama.list') as mock_ollama_list:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Mock ollama.list to raise an exception
            mock_ollama_list.side_effect = Exception("Ollama failure")

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [],
                'downloading': [],
                'model': 'llm'
            })


    def test_get_models_no_config_model(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('ollama.list') as mock_ollama_list:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Create mock models
            model1 = unittest.mock.Mock()
            model1.model = 'llm:latest'
            model1.details = unittest.mock.Mock(parameter_size='7B')
            model1.size = 4100000000
            model1.modified_at = datetime.datetime.fromisoformat('2023-10-01T12:00:00+00:00')
            mock_ollama_list.return_value = {'models': [model1]}

            status, headers, content_bytes = app.request('GET', '/getModels')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {
                'models': [
                    {
                        'id': 'llm:latest',
                        'name': 'llm',
                        'parameters': 7000000000,
                        'size': 4100000000,
                        'modified': '2023-10-01T12:00:00+00:00'
                    }
                ],
                'downloading': []
            })


    def test_download_model_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama_chat.app.DownloadManager') as mock_download_manager:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Initiate model download
            request = {'model': 'llm:latest'}
            status, headers, content_bytes = app.request('POST', '/downloadModel', wsgi_input=json.dumps(request).encode('utf-8'))
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {})

            # Verify DownloadManager was called and stored
            mock_download_manager.assert_called_once_with(app, 'llm:latest')
            self.assertIn('llm:latest', app.downloads)
            self.assertIs(app.downloads['llm:latest'], mock_download_manager.return_value)


    def test_stop_model_download_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'conversations': []}))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Add a mock download to the downloads dictionary
            mock_download = unittest.mock.Mock()
            mock_download.stop = False
            app.downloads['llm:latest'] = mock_download

            # Stop the model download
            request = {'model': 'llm:latest'}
            status, headers, content_bytes = app.request('POST', '/stopModelDownload', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertTrue(mock_download.stop)  # Verify stop flag was set
            self.assertIs(app.downloads['llm:latest'], mock_download)


    def test_stop_model_download_not_downloading(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'conversations': []}))
        ]) as temp_dir:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Try to stop a non-existent download
            request = {'model': 'llm:latest'}
            status, headers, content_bytes = app.request('POST', '/stopModelDownload', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            self.assertDictEqual(app.downloads, {})


    def test_delete_model_success(self):
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('ollama.delete') as mock_ollama_delete:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Delete model 'llm:latest'
            request = {'model': 'llm:latest'}
            status, headers, content_bytes = app.request('POST', '/deleteModel', wsgi_input=json.dumps(request).encode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(json.loads(content_bytes.decode('utf-8')), {})
            mock_ollama_delete.assert_called_once_with('llm:latest')


    @unittest.skipIf(platform.system() != 'Windows', "Skipping this test on non-Windows")
    def test_get_system_info_windows(self): # pragma: no cover
        with create_test_files([
                (('ollama-chat.json',), json.dumps({'conversations': []}))
             ]) as temp_dir, \
             unittest.mock.patch('os.name', 'nt'), \
             unittest.mock.patch('ctypes.windll.kernel32.GlobalMemoryStatusEx') as mock_memory_status:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Mock the MEMORYSTATUSEX structure and the ctypes call
            mock_memory = unittest.mock.Mock()
            mock_memory.ullTotalPhys = 8589934592
            mock_memory_status.return_value = None

            status, headers, content_bytes = app.request('GET', '/getSystemInfo')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'memory': 8589934592})

            # Verify the ctypes call was made with a MEMORYSTATUSEX instance
            self.assertTrue(mock_memory_status.called)
            args, _ = mock_memory_status.call_args
            self.assertIsInstance(args[0], ctypes.Structure)
            self.assertEqual(ctypes.sizeof(args[0]), ctypes.sizeof(MEMORYSTATUSEX))


    @unittest.skipIf(platform.system() == 'Windows', "Skipping this test on Windows")
    def test_get_system_info_unix(self): # pragma: no cover
        with create_test_files([
            (('ollama-chat.json',), json.dumps({'conversations': []}))
        ]) as temp_dir, \
        unittest.mock.patch('os.name', 'posix'), \
        unittest.mock.patch('os.sysconf') as mock_sysconf:
            config_path = os.path.join(temp_dir, 'ollama-chat.json')
            app = OllamaChat(config_path)

            # Mock os.sysconf calls
            mock_sysconf.side_effect = lambda x: {
                'SC_PHYS_PAGES': 2097152,
                'SC_PAGE_SIZE': 4096
            }[x]

            # Make the request
            status, headers, content_bytes = app.request('GET', '/getSystemInfo')
            response = json.loads(content_bytes.decode('utf-8'))
            self.assertEqual(status, '200 OK')
            self.assertListEqual(headers, [('Content-Type', 'application/json')])
            self.assertDictEqual(response, {'memory': 8589934592})  # 8GB in bytes

            # Verify os.sysconf calls
            mock_sysconf.assert_any_call('SC_PHYS_PAGES')
            mock_sysconf.assert_any_call('SC_PAGE_SIZE')
            self.assertEqual(mock_sysconf.call_count, 2)

# {% endraw %}
