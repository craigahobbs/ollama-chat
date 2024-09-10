# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat chat manager
"""

import argparse
import functools
import itertools
import os
import pathlib
import re
import shlex
import threading
import urllib

import ollama


# The ollama chat manager class
class ChatManager():
    __slots__ = ('app', 'conversation_id', 'prompts', 'stop')


    def __init__(self, app, conversation_id, prompts):
        self.app = app
        self.conversation_id = conversation_id
        self.prompts = list(prompts)
        self.stop = False

        # Start the chat thread
        chat_thread = threading.Thread(target=self.chat_thread_fn, args=(self,))
        chat_thread.daemon = True
        chat_thread.start()


    @staticmethod
    def chat_thread_fn(chat):
        try:
            while not chat.stop and chat.prompts:
                # Create the Ollama messages from the conversation
                messages = []
                flags = {}
                with chat.app.config() as config:
                    conversation = config_conversation(config, chat.conversation_id)
                    model = conversation['model']

                    # Add the next user prompt
                    conversation['exchanges'].append({'user': chat.prompts[0], 'model': ''})
                    del chat.prompts[0]

                    # Process user prompt commands append to messages (unless there's a "do" command)
                    for exchange in conversation['exchanges']:
                        flags = {}
                        user_content = _process_commands(exchange['user'], flags)
                        if 'do' not in flags:
                            messages.append({'role': 'user', 'content': user_content})
                            if exchange['model'] != '':
                                messages.append({'role': 'assistant', 'content': exchange['model']})

                    # Help command?
                    if 'help' in flags:
                        exchange = conversation['exchanges'][-1]
                        exchange['model'] = f'```\n{flags["help"].strip()}\n```'
                        continue

                    # Show command?
                    elif 'show' in flags:
                        exchange = conversation['exchanges'][-1]
                        exchange['model'] = user_content
                        continue

                    # Do command?
                    elif 'do' in flags:
                        # Insert the template prompts to the chat
                        template_name, variable_values = flags['do']
                        template = config_template_name(config, template_name)
                        if template is None:
                            raise ValueError(f'unknown template "{template_name}"')
                        _, template_prompts = config_template_prompts(template, variable_values)
                        for template_prompt in reversed(template_prompts):
                            chat.prompts.insert(0, template_prompt)

                        # Update the conversation
                        exchange = conversation['exchanges'][-1]
                        exchange['model'] = f'Executing template "{template_name}"'
                        continue

                # Stream the chat response
                for chunk in ollama.chat(model=model, messages=messages, stream=True):
                    if chat.stop:
                        break

                    # Update the conversation
                    with chat.app.config() as config:
                        conversation = config_conversation(config, chat.conversation_id)
                        exchange = conversation['exchanges'][-1]
                        exchange['model'] += chunk['message']['content']

        except Exception as exc: # pylint: disable=broad-exception-caught
            # Communicate the error
            with chat.app.config() as config:
                conversation = config_conversation(config, chat.conversation_id)
                exchange = conversation['exchanges'][-1]
                exchange['model'] += f'\n**ERROR:** {exc}'

        # Save the conversation
        with chat.app.config(save=True):
            # Delete the application's chat entry
            if chat.conversation_id in chat.app.chats:
                del chat.app.chats[chat.conversation_id]


# Helper to find a conversation by ID
def config_conversation(config, id_):
    return next((conv for conv in config['conversations'] if conv['id'] == id_), None)


# Helper to find a template by name or title
def config_template_name(config, name):
    templates = config['templates'] or []
    return next((tmpl for tmpl in templates if tmpl.get('name') == name or tmpl['title'] == name), None)


# Helper to get the template prompts
def config_template_prompts(template, variable_values):
    title = template['title']
    prompts = template['prompts']
    variables = template.get('variables') or []

    # Missing template variable values?
    for variable in variables:
        if variable['name'] not in variable_values:
            raise ValueError(f'missing variable value for "{variable["name"]}"')

    # Unknown template variable value
    for variable_name in sorted(variable_values.keys()):
        if variable_name not in (variable['name'] for variable in variables):
            raise ValueError(f'unknown variable "{variable_name}"')

    # Render the prompt variables
    if variables:
        re_variables = re.compile(r'\{\{(' + '|'.join(re.escape(variable['name']) for variable in variables) + r')\}\}')
        title = re_variables.sub(lambda match: variable_values[match.group(1)], title)
        prompts = [re_variables.sub(lambda match: variable_values[match.group(1)], prompt) for prompt in prompts]

    return title, prompts


# Process prompt commands
def _process_commands(prompt, flags):
    return _R_COMMAND.sub(functools.partial(_process_commands_sub, flags), prompt)

_R_COMMAND = re.compile(r'^/(?P<cmd>\?|dir|do|file|url)(?P<args> .*)?$', re.MULTILINE)


# Command prompt regex substitution function
def _process_commands_sub(flags, match):
    # Parse command arguments
    command = match.group('cmd')
    argv = [command, *shlex.split(match.group('args') or '')]
    try:
        args = _COMMAND_PARSER.parse_args(args=argv)
    except CommandHelpError as exc:
        flags['help'] = str(exc)
        return f'Displaying help for "{command}" command'

    # Respond with processed prompt?
    if hasattr(args, 'show') and args.show:
        flags['show'] = True

    # Include files from a directory?
    if command == 'dir':
        # Command arguments
        dir_path = str(pathlib.Path(pathlib.PurePosixPath(args.dir)))
        file_exts = set((ext if ext.startswith('.') else f'.{ext}') for ext in itertools.chain([args.ext], args.extra_ext or []))

        # Separate file and directory excludes
        file_excludes = []
        dir_excludes = []
        if args.exclude:
            for exclude in args.exclude:
                (dir_excludes if exclude.endswith('/') else file_excludes).append(exclude)

        # Get the file content
        file_contents = []
        for file_name in sorted(_get_directory_files(dir_path, max(1, args.depth) - 1, file_exts)):
            if file_name in file_excludes or any(file_name.startswith(dir_exclude) for dir_exclude in dir_excludes):
                continue
            with open(file_name, 'r', encoding='utf-8') as fh:
                file_contents.append(_command_file_content(file_name, fh.read()))

        # No files?
        if not file_contents:
            raise ValueError(f'no files found in directory "{args.dir}"')

        # Add file content
        return '\n\n'.join(file_contents)

    # Execute a template by name
    elif command == 'do':
        # Compute the template variable values
        variable_values = {}
        if args.var:
            for variable_name, variable_value in args.var:
                variable_values[variable_name] = variable_value

        # Set the "do" flag
        if 'do' in flags:
            raise ValueError(f'multiple do commands provided ("{flags["do"]}", "{args.name}")')
        flags['do'] = (args.name, variable_values)

        # Add the do-command message
        return f'Executing template "{args.name}"'

    # Include a file?
    elif command == 'file':
        # Command arguments
        file_path = str(pathlib.Path(pathlib.PurePosixPath(args.file)))

        # Add file content
        with open(file_path, 'r', encoding='utf-8') as fh:
            return _command_file_content(file_path, fh.read())

    # Include a URL?
    elif command == 'url':
        # Add URL content
        with urllib.request.urlopen(args.url) as response:
            return _command_file_content(args.url, response.read().decode())

    # Top-level help...
    # elif command == '?':
    flags['help'] = _COMMAND_PARSER.format_help().replace('usage: / ', 'usage: /')
    return 'Displaying top-level help'


# Prompt command argument parser help action
class CommandHelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        raise CommandHelpError(parser.format_help().replace('usage: / ', 'usage: /'))


# Prompt command argument parser help action exception
class CommandHelpError(Exception):
    pass


# Prompt command argument parser
_COMMAND_PARSER = argparse.ArgumentParser(prog='/', add_help=False, exit_on_error=False)
_COMMAND_SUBPARSERS = _COMMAND_PARSER.add_subparsers(dest='command')
_COMMAND_PARSER_HELP = _COMMAND_SUBPARSERS.add_parser('?', add_help=False, exit_on_error=False, help='show prompt command help')
_COMMAND_PARSER_DIR = _COMMAND_SUBPARSERS.add_parser('dir', add_help=False, exit_on_error=False, help='include files from a directory')
_COMMAND_PARSER_DIR.add_argument('dir', help='the directory path')
_COMMAND_PARSER_DIR.add_argument('ext', help='the file extension')
_COMMAND_PARSER_DIR.add_argument('-d', dest='depth', metavar='N', type=int, default=1, help='maximum file recursion depth (default is 1)')
_COMMAND_PARSER_DIR.add_argument('-e', dest='extra_ext', metavar='EXT', action='append', help='additional file extension')
_COMMAND_PARSER_DIR.add_argument('-h', dest='help', action=CommandHelpAction, help='show help')
_COMMAND_PARSER_DIR.add_argument('-n', dest='show', action='store_true', help='respond with user prompt')
_COMMAND_PARSER_DIR.add_argument('-x', dest='exclude', metavar='PATH', action='append', help='exclude file or directory')
_COMMAND_PARSER_DO = _COMMAND_SUBPARSERS.add_parser('do', add_help=False, exit_on_error=False, help='execute a conversation template')
_COMMAND_PARSER_DO.add_argument('name', help='the template name or title')
_COMMAND_PARSER_DO.add_argument('-h', dest='help', action=CommandHelpAction, help='show help')
_COMMAND_PARSER_DO.add_argument('-v', dest='var', metavar=('VAR', 'VAL'), nargs=2, action='append', help='set a template variable')
_COMMAND_PARSER_FILE = _COMMAND_SUBPARSERS.add_parser('file', add_help=False, exit_on_error=False, help='include a file')
_COMMAND_PARSER_FILE.add_argument('file', help='the file path')
_COMMAND_PARSER_FILE.add_argument('-h', dest='help', action=CommandHelpAction, help='show help')
_COMMAND_PARSER_FILE.add_argument('-n', dest='show', action='store_true', help='respond with user prompt')
_COMMAND_PARSER_URL = _COMMAND_SUBPARSERS.add_parser('url', add_help=False, exit_on_error=False, help='include a URL resource')
_COMMAND_PARSER_URL.add_argument('url', help='the resource URL')
_COMMAND_PARSER_URL.add_argument('-h', dest='help', action=CommandHelpAction, help='show help')
_COMMAND_PARSER_URL.add_argument('-n', dest='show', action='store_true', help='respond with user prompt')


# Helper to produce file text content
def _command_file_content(file_name, content):
    content_newline = '\n' if not content.endswith('\n') else ''
    escaped_content = _R_COMMAND_FENCE_ESCAPE.sub(r'\1\```', content)
    return f'**{_escape_markdown_text(file_name)}**\n\n```\n{escaped_content}{content_newline}```'

_R_COMMAND_FENCE_ESCAPE = re.compile(r'^( {0,3})```', re.MULTILINE)


# Helper to escape a string for inclusion in Markdown text
def _escape_markdown_text(text):
    return _RE_ESCAPE_MARKDOWN_TEXT.sub(r'\\\1', text)

_RE_ESCAPE_MARKDOWN_TEXT = re.compile(r'([\\[\]()<>"\'*_~`#=+|-])')


# Helper enumerator to recursively get a directory's files
def _get_directory_files(dir_name, max_depth, file_exts, current_depth=0):
    # Recursion too deep?
    if current_depth > max_depth:
        return

    # Scan the directory for files
    for entry in os.scandir(dir_name):
        if entry.is_file():
            if os.path.splitext(entry.name)[1] in file_exts:
                yield entry.path
        elif entry.is_dir():
            yield from _get_directory_files(entry.path, max_depth, file_exts, current_depth + 1)
