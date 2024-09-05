# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import unittest
import unittest.mock

from ollama_chat.app import OllamaChat


class TestApp(unittest.TestCase):

    def test_init_missing_config(self):
        with unittest.mock.patch('os.path.isfile', return_value=False) as mock_isfile:
            app = OllamaChat('ollama-chat.json')
            self.assertEqual(app.config.config_path, 'ollama-chat.json')
            self.assertDictEqual(app.chats, {})
            self.assertListEqual(sorted(app.requests.keys()), [
                'chisel_doc',
                'chisel_doc_index',
                'chisel_doc_request',
                'createTemplateFromConversation',
                'deleteConversation',
                'deleteConversationExchange',
                'getConversation',
                'getConversations',
                'getTemplate',
                'index.html',
                'ollamaChat.bare',
                'ollamaChatConversation.bare',
                'redirect_doc',
                'regenerateConversationExchange',
                'replyConversation',
                'setConversationTitle',
                'setModel',
                'startConversation',
                'startTemplate',
                'stopConversation'
            ])

        mock_isfile.assert_called_once_with('ollama-chat.json')
