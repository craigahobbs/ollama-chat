import datetime
from html.parser import HTMLParser
import http.client
import json
import os
import re
import sys
import time
import urllib.error
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

    # X minutes/hours/days/weeks/months/years ago?
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
        elif unit == 'year':
            return today - datetime.timedelta(days=count * 365)
        else:
            return today

    # about a hour/minute ago?
    elif _regex_about.match(modified):
        return today

    raise ValueError(f'Unrecognized modified string: {modified}')

_regex_modified_ago = re.compile(r'^(?P<count>\d+)\s+(?P<unit>minute|hour|day|week|month|year)s?\s+ago$')
_regex_about = re.compile(r'^about\s+an?\s+(hour|minute)\s+ago$')


def _parse_count(count, model_name):
    try:
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
    except:
        print(f'Info: "{model_name}" has invalid size "{count}"', file=sys.stderr)
        return 0

_regex_count_moe = re.compile(r'^(?P<mult>[1-9]\d*)x')


# Fetch a URL's text, retrying transient failures with exponential backoff (the many sequential
# fetches can be throttled). Returns None if the page no longer exists. A persistent failure
# raises, failing the run - stale-but-complete model data beats publishing incomplete data.
def _fetch(url):
    attempt = 0
    while True:
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request) as response:
                return response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return None
            if attempt == _FETCH_ATTEMPTS - 1:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == _FETCH_ATTEMPTS - 1:
                raise
        time.sleep(_FETCH_RETRY_SECONDS * 2 ** attempt)
        attempt += 1

_FETCH_ATTEMPTS = 5
_FETCH_RETRY_SECONDS = 5


# Scrape a model's tags web page for its cloud and MLX variant tags (e.g. "cloud", "31b-cloud", "12b-mlx").
# Quantization tags (e.g. "12b-mlx-bf16", "12b-it-q4_K_M") are not variants and are excluded.
# Returns None if the model's tags page no longer exists.
def _scrape_variant_tags(model_name):
    html = _fetch(f'https://ollama.com/library/{model_name}/tags')
    if html is None:
        return None
    tags = dict.fromkeys(re.findall(rf'href="/library/{re.escape(model_name)}:([^"]+)"', html))
    cloud_tags = [tag for tag in tags if tag == 'cloud' or tag.endswith('-cloud')]
    mlx_tags = [tag for tag in tags if tag == 'mlx' or tag.endswith('-mlx')]

    # Drop the bare alias tag when sized tags exist (e.g. "cloud" alongside "31b-cloud")
    if 'cloud' in cloud_tags and len(cloud_tags) > 1:
        cloud_tags.remove('cloud')
    if 'mlx' in mlx_tags and len(mlx_tags) > 1:
        mlx_tags.remove('mlx')

    return cloud_tags, mlx_tags


# Is the size a cloud/MLX variant tag? The model's tags page is the source of truth for those.
def _is_cloud_mlx_size(size):
    return size in ('cloud', 'mlx') or size.endswith('-cloud') or size.endswith('-mlx')


# Parse the parameter count from a cloud/MLX variant tag (e.g. "31b-cloud" -> 31e9).
# A bare "cloud"/"mlx" tag has an unknown parameter count, zero.
def _parse_tag_count(tag, kind, model_name):
    size = tag[:-len(kind) - 1] if tag.endswith(f'-{kind}') else ''
    return _parse_count(size, model_name) if size else 0


def main():
    # Fetch HTML
    html = _fetch('https://ollama.com/library')

    # Parse HTML and extract descriptions
    parser = OllamaModelParser()
    parser.feed(html)
    raw_models = parser.models

    # Parse scraped model info values
    models = []
    for model_name, raw_model in raw_models.items():
        # Scrape the model's cloud and MLX variant tags
        variant_tags = _scrape_variant_tags(model_name)
        if variant_tags is None:
            print(f'Warning: "{model_name}" no longer exists', file=sys.stderr)
            continue
        cloud_tags, mlx_tags = variant_tags

        # No model variants?
        local_sizes = [size for size in raw_model['sizes'] if not _is_cloud_mlx_size(size)]
        if not local_sizes and not cloud_tags and not mlx_tags:
            print(f'Warning: "{model_name}" has no sizes', file=sys.stderr)
            continue

        # Create the model variants - local sizes, then MLX, then cloud
        variants = [
            {
                'id': f'{model_name}:{size}',
                'size': size,
                'parameters': _parse_count(size, model_name)
            }
            for size in local_sizes
        ]
        for kind, kind_tags in (('mlx', mlx_tags), ('cloud', cloud_tags)):
            for tag in kind_tags:
                variants.append({
                    'id': f'{model_name}:{tag}',
                    'size': tag,
                    'parameters': _parse_tag_count(tag, kind, model_name),
                    kind: True
                })

        models.append({
            'name': model_name,
            'description': raw_model.get('description', 'No model description provided.'),
            'modified': _parse_modified(raw_model['modified']).isoformat(),
            'downloads': _parse_count(raw_model['downloads'], model_name),
            'variants': variants
        })

    # Validate the model JSON
    ollama_chat_smd = 'ollamaChat.smd'
    if not os.path.isfile(ollama_chat_smd):
        ollama_chat_smd = '../../src/ollama_chat/static/ollamaChat.smd'
    with open(ollama_chat_smd, 'r', encoding='utf-8') as fh:
        ollama_chat_types = schema_markdown.parse_schema_markdown(fh.read())
    schema_markdown.validate_type(ollama_chat_types, 'OllamaChatModels', models)

    # Output the model JSON
    print(json.dumps(sorted(models, key=lambda model: model['name']), indent=4))


if __name__ == '__main__':
    main()
