# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import unittest

import ollama_chat.__main__


class TestMain(unittest.TestCase):

    def test_main_submodule(self):
        self.assertTrue(ollama_chat.__main__)
