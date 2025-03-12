# {% raw %}
# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import base64
import unittest
import unittest.mock

from ollama_chat.chat import _escape_markdown_text, _process_commands, config_template_prompts

from .util import create_test_files


class TestTemplatePrompts(unittest.TestCase):

    def test_config_template_prompts_basic(self):
        template = {
            'title': 'Simple Template',
            'prompts': ['Hello world', 'Second prompt']
        }
        variable_values = {}
        title, prompts = config_template_prompts(template, variable_values)
        self.assertEqual(title, 'Simple Template')
        self.assertListEqual(prompts, ['Hello world', 'Second prompt'])


    def test_config_template_prompts_with_variables(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}', 'How are you {{name}}?'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {'name': 'Alice'}
        title, prompts = config_template_prompts(template, variable_values)
        self.assertEqual(title, 'Hello Alice')
        self.assertListEqual(prompts, ['Greetings Alice', 'How are you Alice?'])


    def test_config_template_prompts_missing_variable(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {}
        with self.assertRaises(ValueError) as context:
            config_template_prompts(template, variable_values)
        self.assertEqual(str(context.exception), 'missing variable value for "name"')


    def test_config_template_prompts_unknown_variable(self):
        template = {
            'title': 'Hello {{name}}',
            'prompts': ['Greetings {{name}}'],
            'variables': [{'name': 'name'}]
        }
        variable_values = {'name': 'Alice', 'age': '30'}
        with self.assertRaises(ValueError) as context:
            config_template_prompts(template, variable_values)
        self.assertEqual(str(context.exception), 'unknown variable "age"')


    def test_config_template_prompts_multiple_variables(self):
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

    def test_process_commands_no_commands(self):
        flags = {}
        self.assertEqual(_process_commands('Hello, how are you?', flags), 'Hello, how are you?')
        self.assertDictEqual(flags, {})


    def test_process_commands_help_top(self):
        flags = {}
        self.assertEqual(_process_commands('/?', flags), 'Displaying top-level help')
        self.assertTrue(flags['help'].startswith('usage: /{?,dir,do,file,image,url}'))


    def test_process_commands_help(self):
        flags = {}
        self.assertEqual(_process_commands('/file test.txt -h', flags), 'Displaying help for "file" command')
        self.assertTrue(flags['help'].startswith('usage: /file [-h] [-n] file'))


    def test_process_commands_dir(self):
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
```
Test 1
```
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_process_commands_dir_depth(self):
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
```
Test 2
```
</ {_escape_markdown_text(temp_dir)}/subdir/test2.txt>

<{_escape_markdown_text(temp_dir)}/test.txt>
```
Test 1
```
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_process_commands_do(self):
        flags = {}
        self.assertEqual(_process_commands('/do template_name -v var1 val1', flags), 'Executing template "template_name"')
        self.assertDictEqual(flags, {'do': [('template_name', {'var1': 'val1'})]})


    def test_process_commands_file(self):
        with create_test_files([
            (('test.txt',), 'file content')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_dir}/test.txt', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/test.txt>
```
file content
```
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {})


    def test_process_commands_file_show(self):
        with create_test_files([
            (('test.txt',), 'file content')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(
                _process_commands(f'/file {temp_dir}/test.txt -n', flags),
                f'''\
<{_escape_markdown_text(temp_dir)}/test.txt>
```
file content
```
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertDictEqual(flags, {'show': True})


    def test_process_commands_image(self):
        with create_test_files([
            (('test.jpg',), 'image data')
        ]) as temp_dir:
            flags = {}
            self.assertEqual(_process_commands(f'/image {temp_dir}/test.jpg', flags), '')
            self.assertDictEqual(flags, {'images': [base64.b64encode(b'image data').decode('utf-8')]})


    def test_process_commands_url(self):
        with unittest.mock.patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = unittest.mock.Mock()
            mock_response.read.return_value = b'url content'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            flags = {}
            self.assertEqual(
                _process_commands('/url http://example.com', flags),
                '''\
<http://example.com>
```
url content
```
</ http://example.com>'''
            )
            self.assertDictEqual(flags, {})


    def test_process_commands_multiple(self):
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
```
file content
```
</ {_escape_markdown_text(temp_dir)}/test.txt>'''
            )
            self.assertIn('help', flags)


    def test_process_commands_file_error(self):
        with create_test_files([]) as temp_dir:
            flags = {}
            with self.assertRaises(FileNotFoundError):
                _process_commands(f'/file {temp_dir}/nonexistent.txt', flags)

# {% endraw %}
