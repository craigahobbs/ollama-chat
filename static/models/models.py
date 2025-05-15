import datetime
from html.parser import HTMLParser
import json
import os
import re
import urllib.request

import schema_markdown


# Scrape the local model metadata from the Ollama models web page
class OllamaModelParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.models = {}
        self.current_model = None
        self.current_field = None
        self.paragraph_count = 0


    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # Handle model name from URL
        if tag == 'a' and not self.current_model and 'href' in attrs and attrs['href'].startswith('/library/'):
            self.current_model = attrs['href'].split('/')[-1].lower()
            if self.current_model not in self.models:
                self.models[self.current_model] = {'sizes': []}
            self.paragraph_count = 0

        # Handle description - track first paragraph after model start
        elif tag == 'p' and self.current_model:
            self.paragraph_count += 1
            if self.paragraph_count == 1:
                self.current_field = 'description'

        # Handle spans with x-test attributes
        elif tag == 'span' and self.current_model:
            if 'x-test-size' in attrs:
                self.current_field = 'sizes'
            elif 'x-test-pull-count' in attrs:
                self.current_field = 'downloads'
            elif 'x-test-updated' in attrs:
                self.current_field = 'modified'


    def handle_endtag(self, tag):
        if tag in ('p', 'span'):
            self.current_field = None
        elif tag == 'a':
            self.current_model = None
            self.paragraph_count = 0


    def handle_data(self, data):
        if self.current_field:
            model_info = self.models[self.current_model]
            model_value = model_info.get(self.current_field)
            if isinstance(model_value, list):
                model_info[self.current_field].append(data.strip())
            else:
                model_info[self.current_field] = data.strip()


def _parse_modified(modified):
    today = datetime.date.today()

    # Just now?
    if modified in ('just now', 'an hour ago'):
        return today

    # Yesterday?
    if modified == 'yesterday':
        return (today - datetime.timedelta(days=1))

    # X minutes/hours/days/weeks/months ago?
    m_ago = _regex_modified_ago.match(modified)
    if m_ago:
        count = int(m_ago.group('count'))
        unit = m_ago.group('unit')
        if unit == 'day':
            return today - datetime.timedelta(days=count)
        elif unit == 'week':
            return today - datetime.timedelta(weeks=count)
        elif unit == 'month':
            return today - datetime.timedelta(days=count * 30)
        else:
            return today

    # about a hour/minute ago?
    elif _regex_about.match(modified):
        return today

    raise ValueError(f'Unrecognized modified string: {modified}')

_regex_modified_ago = re.compile(r'^(?P<count>\d+)\s+(?P<unit>minute|hour|day|week|month|year)s?\s+ago$')
_regex_about = re.compile(r'^about\s+an?\s+(hour|minute)\s+ago$')


def _parse_count(count):
    # Mixture of experts?
    multiplier = 1
    m_moe = _regex_count_moe.match(count)
    if m_moe:
        multiplier = int(m_moe.group('mult'))
        count = count[m_moe.end():]

    # Parse the count
    unit = count[-1].lower()
    if unit == 'b':
        return int(multiplier * float(count[:-1]) * 1e9)
    elif unit == 'm':
        return int(multiplier * float(count[:-1]) * 1e6)
    elif unit == 'k':
        return int(multiplier * float(count[:-1]) * 1e3)
    return multiplier * int(count.replace(',', ''))

_regex_count_moe = re.compile(r'^(?P<mult>[1-9]\d*)x')


def main():
    # Fetch HTML
    request = urllib.request.Request('https://ollama.com/library')
    with urllib.request.urlopen(request) as response:
        html = response.read().decode('utf-8')

    # Parse HTML and extract descriptions
    parser = OllamaModelParser()
    parser.feed(html)
    raw_models = parser.models

    # Parse scraped model info values
    models = []
    for model_name, raw_model in raw_models.items():
        if raw_model['sizes']:
            models.append({
                'name': model_name,
                'description': raw_model['description'],
                'modified': _parse_modified(raw_model['modified']).isoformat(),
                'downloads': _parse_count(raw_model['downloads']),
                'variants': [
                    {
                        'id': f'{model_name}:{size}',
                        'size': size,
                        'parameters': _parse_count(size)
                    }
                    for size in raw_model['sizes']
                ]
            })

    # Validate the model JSON
    ollama_chat_smd = 'ollamaChat.smd'
    if not os.path.isfile(ollama_chat_smd):
        ollama_chat_smd = 'src/ollama_chat/static/ollamaChat.smd'
    with open(ollama_chat_smd, 'r', encoding='utf-8') as fh:
        ollama_chat_types = schema_markdown.parse_schema_markdown(fh.read())
    schema_markdown.validate_type(ollama_chat_types, 'OllamaChatModels', models)

    # Output the model JSON
    print(json.dumps(sorted(models, key=lambda model: model['name']), indent=4))


if __name__ == '__main__':
    main()
