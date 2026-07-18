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
        self.in_description = False
        self.span_is_size = False
        self.span_text = None
        self.prior_span = None


    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # Handle model name from URL - each model's entire entry is inside its anchor
        # (assumes model entries never contain nested anchors)
        if tag == 'a' and not self.current_model and attrs.get('href', '').startswith('/library/'):
            self.current_model = attrs['href'].split('/')[-1].lower()
            if self.current_model not in self.models:
                self.models[self.current_model] = {'sizes': []}

        # Handle description - the "break-words" paragraph
        elif tag == 'p' and self.current_model and 'break-words' in attrs.get('class', ''):
            self.in_description = True

        elif tag == 'span' and self.current_model:
            # Model sizes are the blue "pill" spans
            self.span_is_size = 'text-blue-600' in attrs.get('class', '')
            self.span_text = ''

            # The updated span's title attribute is the exact updated timestamp - ignore other
            # titled spans (e.g. tooltips). A missing timestamp fails loudly in main.
            title = attrs.get('title', '')
            if title.endswith(' UTC'):
                self.models[self.current_model]['modified'] = title


    def handle_endtag(self, tag):
        if tag == 'p':
            self.in_description = False
        elif tag == 'a':
            self.current_model = None
            self.prior_span = None
        elif tag == 'span' and self.current_model and self.span_text is not None:
            text = self.span_text.strip()
            if self.span_is_size and text:
                self.models[self.current_model]['sizes'].append(text)
            elif text == 'Pulls':
                # The pull count span immediately precedes its "Pulls" label span
                self.models[self.current_model]['downloads'] = self.prior_span
            elif text:
                self.prior_span = text
            self.span_is_size = False
            self.span_text = None


    def handle_data(self, data):
        if self.span_text is not None:
            self.span_text += data
        elif self.in_description and self.current_model:
            # Accumulate - inline markup splits the text into multiple data chunks
            model_info = self.models[self.current_model]
            model_info['description'] = model_info.get('description', '') + data


# Parse an updated span title attribute timestamp (e.g. "Nov 30, 2024 10:34 PM UTC")
def _parse_modified(modified):
    return datetime.datetime.strptime(modified, '%b %d, %Y %I:%M %p UTC').date()


def _parse_count(count, model_name):
    original = count
    try:
        # Mixture of experts?
        multiplier = 1
        m_moe = _regex_count_moe.match(count)
        if m_moe:
            multiplier = int(m_moe.group('mult'))
            count = count[m_moe.end():]

        # Effective parameter count? (e.g. gemma3n's "e2b" - the model's effective footprint)
        if _regex_count_effective.match(count):
            count = count[1:]

        # Parse the count
        scale = _COUNT_UNIT_SCALES.get(count[-1].lower())
        if scale:
            return int(multiplier * float(count[:-1]) * scale)
        return multiplier * int(count.replace(',', ''))
    except (ValueError, IndexError):
        print(f'Info: "{model_name}" has invalid size "{original}"', file=sys.stderr)
        return 0

_regex_count_moe = re.compile(r'^(?P<mult>[1-9]\d*)x')
_regex_count_effective = re.compile(r'^e\d')
_COUNT_UNIT_SCALES = {'t': 1e12, 'b': 1e9, 'm': 1e6, 'k': 1e3}


# Fetch a URL's text, retrying transient failures with exponential backoff (the many sequential
# fetches can be throttled). Returns None if the page no longer exists. A persistent failure
# raises, failing the run - stale-but-complete model data beats publishing incomplete data.
def _fetch(url):
    attempt = 0
    while True:
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
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
_FETCH_TIMEOUT_SECONDS = 30


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
    size = '' if tag == kind else tag.removesuffix(f'-{kind}')
    return _parse_count(size, model_name) if size else 0


def main():
    # Fetch HTML
    html = _fetch('https://ollama.com/library')

    # Parse the library page model entries
    parser = OllamaModelParser()
    parser.feed(html)
    parser.close()
    raw_models = parser.models

    # Parse scraped model info values
    models = []
    skipped = 0
    for model_name, raw_model in raw_models.items():
        # Scrape the model's cloud and MLX variant tags
        variant_tags = _scrape_variant_tags(model_name)
        if variant_tags is None:
            print(f'Warning: "{model_name}" no longer exists', file=sys.stderr)
            skipped += 1
            continue
        cloud_tags, mlx_tags = variant_tags

        # No model variants?
        local_sizes = [size for size in dict.fromkeys(raw_model['sizes']) if not _is_cloud_mlx_size(size)]
        if not local_sizes and not cloud_tags and not mlx_tags:
            print(f'Warning: "{model_name}" has no sizes', file=sys.stderr)
            skipped += 1
            continue

        # An unparseable pull count means the scraper is broken, not the data
        downloads = _parse_count(raw_model['downloads'], model_name)
        if not downloads:
            raise ValueError(f'"{model_name}" has invalid downloads "{raw_model["downloads"]}"')

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
            'description': raw_model.get('description', '').strip() or _DEFAULT_DESCRIPTION,
            'modified': _parse_modified(raw_model['modified']).isoformat(),
            'downloads': downloads,
            'variants': variants
        })

    # Sanity-check the scrape's completeness - a partial scraping failure (e.g. a page redesign)
    # must fail the run loudly rather than publish silently degraded model data
    if len(models) < _SANITY_MIN_MODELS:
        raise ValueError(f'Suspiciously few models scraped ({len(models)} < {_SANITY_MIN_MODELS})')
    if skipped > _SANITY_MAX_SKIPPED * len(raw_models):
        raise ValueError(f'Suspiciously many models skipped ({skipped} of {len(raw_models)})')
    descriptionless = sum(1 for model in models if model['description'] == _DEFAULT_DESCRIPTION)
    if descriptionless > _SANITY_MAX_DESCRIPTIONLESS * len(models):
        raise ValueError(f'Suspiciously many models without descriptions ({descriptionless} of {len(models)})')
    for kind in ('cloud', 'mlx'):
        if not any(variant.get(kind) for model in models for variant in model['variants']):
            raise ValueError(f'No {kind} model variants scraped')

    # Validate the model JSON
    ollama_chat_smd = 'ollamaChat.smd'
    if not os.path.isfile(ollama_chat_smd):
        ollama_chat_smd = '../../src/ollama_chat/static/ollamaChat.smd'
    with open(ollama_chat_smd, 'r', encoding='utf-8') as fh:
        ollama_chat_types = schema_markdown.parse_schema_markdown(fh.read())
    schema_markdown.validate_type(ollama_chat_types, 'OllamaChatModels', models)

    # Output the model JSON
    print(json.dumps(sorted(models, key=lambda model: model['name']), indent=4))

_DEFAULT_DESCRIPTION = 'No model description provided.'
_SANITY_MIN_MODELS = 150
_SANITY_MAX_SKIPPED = 0.2
_SANITY_MAX_DESCRIPTIONLESS = 0.1


if __name__ == '__main__':
    main()
