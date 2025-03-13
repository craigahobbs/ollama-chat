# {% raw %}
# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import json
import os
import unittest
import unittest.mock

from schema_markdown import encode_query_string
from ollama_chat.app import OllamaChat

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

            # Check config remains unchanged
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

            # Check config remains unchanged
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

            # Check config remains unchanged
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

# {% endraw %}
