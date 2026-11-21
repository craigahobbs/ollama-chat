# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import importlib.util
from io import BytesIO, StringIO
import json
import os
import runpy
import unittest
import unittest.mock
import urllib.error


# The models scraper script's helpers are module-private
# pylint: disable=protected-access


# Load the models scraper script (static/models/models.py is not part of the package)
MODELS_PY = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'models', 'models.py')
_MODELS_SPEC = importlib.util.spec_from_file_location('ollama_chat_models', MODELS_PY)
models = importlib.util.module_from_spec(_MODELS_SPEC)
_MODELS_SPEC.loader.exec_module(models)


# Helper to create an Ollama library page model entry
def library_entry(name, description='A model', sizes=(), capabilities=(), pulls='1.5M',
                  updated='Nov 30, 2024 10:34 PM UTC'):
    pills = ''.join(
        f'<span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">{capability}</span>' for capability in capabilities
    )
    pills += ''.join(f'<span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">{size}</span>' for size in sizes)
    description_html = f'<p class="max-w-lg break-words text-md">{description}</p>' if description is not None else ''
    return f'''\
<li>
  <a href="/library/{name}" class="group w-full">
    <div title="{name}" class="flex flex-col">
      <h2><div><span class="group-hover:underline truncate">{name}</span></div></h2>
      {description_html}
    </div>
    <div class="flex flex-col"><div class="flex flex-wrap">{pills}</div>
      <p class="flex">
        <span class="flex items-center"><svg></svg><span>{pulls}</span><span class="hidden sm:flex">&nbsp;Pulls</span></span>
        <span class="flex items-center"><span>3</span><span class="hidden sm:flex">&nbsp;Tags</span></span>
        <span class="flex items-center" title="{updated}"><span>Updated&nbsp;</span><span>19 hours ago</span></span>
      </p>
    </div>
  </a>
</li>
'''


# Helper to create an Ollama library page
def library_page(*entries):
    return f'<html><body><a href="/">Home</a><a href="/library">Models</a><ul>{"".join(entries)}</ul></body></html>'


# Helper to create an Ollama model tags page
def tags_page(name, *tags):
    return '<html><body>' + ''.join(f'<a href="/library/{name}:{tag}">{name}:{tag}</a>' for tag in tags) + '</body></html>'


# Helper to create the Ollama registry responses for a model's "latest" tag config
def registry_responses(name, config):
    registry_url = f'https://registry.ollama.ai/v2/library/{name}'
    return {
        f'{registry_url}/manifests/latest': json.dumps({'config': {'digest': f'sha256:{name}'}}),
        f'{registry_url}/blobs/sha256:{name}': json.dumps(config)
    }


# Helper to create a fake _fetch function from a URL-to-response map (missing URLs return None)
def fake_fetch(responses):
    return unittest.mock.Mock(side_effect=responses.get)


class TestOllamaModelParser(unittest.TestCase):

    def test_parser(self):
        parser = models.OllamaModelParser()
        parser.feed(library_page(
            library_entry('llama3', 'Meta&#39;s <b>Llama</b> model', sizes=('8b', '70b'), capabilities=('tools',)),
            library_entry('LLaVA', None, capabilities=('vision',), pulls='12.8K', updated='Jan 2, 2025 1:05 AM UTC')
        ))
        parser.close()
        self.assertDictEqual(parser.models, {
            'llama3': {
                'sizes': ['8b', '70b'],
                'capabilities': ['tools'],
                'description': "Meta's Llama model",
                'downloads': '1.5M',
                'modified': 'Nov 30, 2024 10:34 PM UTC'
            },
            'llava': {
                'sizes': [],
                'capabilities': ['vision'],
                'downloads': '12.8K',
                'modified': 'Jan 2, 2025 1:05 AM UTC'
            }
        })


    def test_parser_empty_spans(self):
        parser = models.OllamaModelParser()
        parser.feed(
            '<a href="/library/llama3"><span class="text-blue-600"> </span><span class="text-indigo-600"></span>'
            '<span></span><span>42</span><span>Pulls</span></a><span>outside</span>'
            # A repeated model entry accumulates into the first
            '<a href="/library/llama3"><span class="text-blue-600">8b</span></a>'
        )
        parser.close()
        self.assertDictEqual(parser.models, {'llama3': {'sizes': ['8b'], 'capabilities': [], 'downloads': '42'}})


class TestParse(unittest.TestCase):

    def test_parse_modified(self):
        self.assertEqual(models._parse_modified('Nov 30, 2024 10:34 PM UTC').isoformat(), '2024-11-30')


    def test_parse_count(self):
        with unittest.mock.patch('sys.stderr', StringIO()) as stderr:
            self.assertEqual(models._parse_count('8b', 'test'), 8000000000)
            self.assertEqual(models._parse_count('1.5M', 'test'), 1500000)
            self.assertEqual(models._parse_count('12.8K', 'test'), 12800)
            self.assertEqual(models._parse_count('1.02T', 'test'), 1020000000000)
            self.assertEqual(models._parse_count('891.14M', 'test'), 891140000)
            self.assertEqual(models._parse_count('42', 'test'), 42)
            self.assertEqual(models._parse_count('1,234', 'test'), 1234)
            self.assertEqual(models._parse_count('8x7b', 'test'), 56000000000)
            self.assertEqual(models._parse_count('e2b', 'test'), 2000000000)
            self.assertEqual(models._parse_count('30b-a3b', 'test'), 30000000000)
            self.assertEqual(models._parse_count('26B-A5.5B', 'test'), 26000000000)
            self.assertEqual(models._parse_count('8x7b-a13b', 'test'), 56000000000)
        self.assertEqual(stderr.getvalue(), '')


    def test_parse_count_invalid(self):
        with unittest.mock.patch('sys.stderr', StringIO()) as stderr:
            self.assertEqual(models._parse_count('latest', 'test'), 0)
            self.assertEqual(models._parse_count('', 'test'), 0)
        self.assertEqual(stderr.getvalue(), 'Info: "test" has invalid size "latest"\nInfo: "test" has invalid size ""\n')


    def test_parse_description_count(self):
        self.assertEqual(models._parse_description_count('A small model: 26B parameters, 5.4B active.', 'test'), 26000000000)
        self.assertEqual(models._parse_description_count('OpenHermes 2.5 is a 7B model fine-tuned.', 'test'), 7000000000)
        self.assertEqual(models._parse_description_count('A 33B total parameter MoE model.', 'test'), 33000000000)
        self.assertEqual(models._parse_description_count('A 30B Mixture-of-Experts model.', 'test'), 30000000000)
        self.assertEqual(models._parse_description_count('A 421M decision model.', 'test'), 421000000)
        self.assertEqual(models._parse_description_count('With 3B active parameters and 30B parameters.', 'test'), 30000000000)
        self.assertEqual(models._parse_description_count('Two Llama 2 70B models combined.', 'test'), 0)
        self.assertEqual(models._parse_description_count('The strongest model in the 30B class.', 'test'), 0)


    def test_is_cloud_mlx_size(self):
        self.assertTrue(models._is_cloud_mlx_size('cloud'))
        self.assertTrue(models._is_cloud_mlx_size('mlx'))
        self.assertTrue(models._is_cloud_mlx_size('31b-cloud'))
        self.assertTrue(models._is_cloud_mlx_size('12b-mlx'))
        self.assertFalse(models._is_cloud_mlx_size('12b'))
        self.assertFalse(models._is_cloud_mlx_size('12b-mlx-bf16'))


    def test_is_sized_tag(self):
        for tag in ('8b', '1.5B', '30b-a3b', '8x7b', 'e2b', '40m-en', '40m-multilingual', '7b-v2.5', '26b-a5b'):
            self.assertTrue(models._is_sized_tag(tag), tag)
        for tag in (
            'latest', 'v2.5', 'en', 'bf16', '8b-cloud', '8b-mlx', '8b-q4_K_M', '8b-iq2_xs', '8b-instruct-fp16', '8b-f16',
            '0.8b-mlx-bf16', '26b-a5b-nvfp4', '30b-mxfp8', '7b-int4'
        ):
            self.assertFalse(models._is_sized_tag(tag), tag)


    def test_parse_size_count(self):
        with unittest.mock.patch('sys.stderr', StringIO()) as stderr:
            self.assertEqual(models._parse_size_count('8b', 'test'), 8000000000)
            self.assertEqual(models._parse_size_count('30b-a3b', 'test'), 30000000000)
            self.assertEqual(models._parse_size_count('40m-en', 'test'), 40000000)
            self.assertEqual(models._parse_size_count('7b-v2.5', 'test'), 7000000000)
            self.assertEqual(models._parse_size_count('latest', 'test'), 0)
        self.assertEqual(stderr.getvalue(), 'Info: "test" has invalid size "latest"\n')


    def test_parse_tag_count(self):
        self.assertEqual(models._parse_tag_count('31b-cloud', 'cloud', 'test'), 31000000000)
        self.assertEqual(models._parse_tag_count('12b-mlx', 'mlx', 'test'), 12000000000)
        self.assertEqual(models._parse_tag_count('cloud', 'cloud', 'test'), 0)


class TestFetch(unittest.TestCase):

    @staticmethod
    def response(text):
        return BytesIO(text.encode('utf-8'))


    # Create an HTTP error - closed at test cleanup (an unclosed HTTPError warns when collected)
    def http_error(self, code):
        error = urllib.error.HTTPError('https://ollama.com/library', code, 'Error', {}, BytesIO())
        self.addCleanup(error.close)
        return error


    def test_fetch(self):
        with unittest.mock.patch('urllib.request.urlopen', return_value=self.response('Hello')) as mock_urlopen, \
             unittest.mock.patch('time.sleep') as mock_sleep:
            self.assertEqual(models._fetch('https://ollama.com/library'), 'Hello')
        self.assertEqual(mock_urlopen.call_count, 1)
        request, = mock_urlopen.call_args.args
        self.assertEqual(request.full_url, 'https://ollama.com/library')
        self.assertDictEqual(mock_urlopen.call_args.kwargs, {'timeout': 30})
        mock_sleep.assert_not_called()


    def test_fetch_not_found(self):
        for code in (404, 410):
            with unittest.mock.patch('urllib.request.urlopen', side_effect=self.http_error(code)) as mock_urlopen, \
                 unittest.mock.patch('time.sleep') as mock_sleep:
                self.assertIsNone(models._fetch('https://ollama.com/library'))
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_sleep.assert_not_called()


    def test_fetch_retry(self):
        side_effect = [self.http_error(503), OSError('Connection reset'), self.response('Hello')]
        with unittest.mock.patch('urllib.request.urlopen', side_effect=side_effect) as mock_urlopen, \
             unittest.mock.patch('time.sleep') as mock_sleep:
            self.assertEqual(models._fetch('https://ollama.com/library'), 'Hello')
        self.assertEqual(mock_urlopen.call_count, 3)
        self.assertListEqual(mock_sleep.call_args_list, [unittest.mock.call(5), unittest.mock.call(10)])


    def test_fetch_http_error(self):
        with unittest.mock.patch('urllib.request.urlopen', side_effect=self.http_error(500)) as mock_urlopen, \
             unittest.mock.patch('time.sleep') as mock_sleep:
            with self.assertRaises(urllib.error.HTTPError):
                models._fetch('https://ollama.com/library')
        self.assertEqual(mock_urlopen.call_count, 5)
        self.assertListEqual(mock_sleep.call_args_list, [unittest.mock.call(seconds) for seconds in (5, 10, 20, 40)])


    def test_fetch_os_error(self):
        with unittest.mock.patch('urllib.request.urlopen', side_effect=OSError('Connection reset')) as mock_urlopen, \
             unittest.mock.patch('time.sleep') as mock_sleep:
            with self.assertRaises(OSError):
                models._fetch('https://ollama.com/library')
        self.assertEqual(mock_urlopen.call_count, 5)
        self.assertEqual(mock_sleep.call_count, 4)


class TestScrape(unittest.TestCase):

    def test_scrape_variant_tags(self):
        html = tags_page('qwen', 'latest', '8b', '8b-q4_K_M', 'cloud', '235b-cloud', 'mlx', '8b-mlx', '8b-mlx-bf16', '8b')
        with unittest.mock.patch.object(models, '_fetch', fake_fetch({'https://ollama.com/library/qwen/tags': html})):
            self.assertTupleEqual(models._scrape_variant_tags('qwen'), (['235b-cloud'], ['8b-mlx'], ['8b'], True))


    def test_scrape_variant_tags_aliases(self):
        html = tags_page('kimi', 'cloud', 'mlx', 'kimi-other:latest')
        with unittest.mock.patch.object(models, '_fetch', fake_fetch({'https://ollama.com/library/kimi/tags': html})):
            self.assertTupleEqual(models._scrape_variant_tags('kimi'), (['cloud'], ['mlx'], [], False))


    def test_scrape_variant_tags_missing(self):
        with unittest.mock.patch.object(models, '_fetch', fake_fetch({})):
            self.assertIsNone(models._scrape_variant_tags('gone'))


    def test_scrape_latest_count(self):
        with unittest.mock.patch.object(models, '_fetch', fake_fetch(registry_responses('kev', {'model_type': '9.0B'}))):
            self.assertEqual(models._scrape_latest_count('kev'), 9000000000)


    def test_scrape_latest_count_unavailable(self):
        registry_url = 'https://registry.ollama.ai/v2/library/test'
        manifest = json.dumps({'config': {'digest': 'sha256:test'}})
        for responses in (
            # No manifest
            {},
            # No config blob
            {f'{registry_url}/manifests/latest': manifest},
            # Invalid manifest JSON
            {f'{registry_url}/manifests/latest': 'not JSON'},
            # Manifest without a config digest
            {f'{registry_url}/manifests/latest': json.dumps({'layers': []})},
            # Config isn't an object
            {f'{registry_url}/manifests/latest': manifest, f'{registry_url}/blobs/sha256:test': '[]'},
            # No model type
            registry_responses('test', {'model_format': 'safetensors'}),
            # Empty model type
            registry_responses('test', {'model_type': ''}),
            # Non-string model type
            registry_responses('test', {'model_type': 9})
        ):
            with unittest.mock.patch.object(models, '_fetch', fake_fetch(responses)):
                self.assertEqual(models._scrape_latest_count('test'), 0)


class TestMain(unittest.TestCase):

    # Run the scraper's main with fake fetch responses - returns the output model JSON and stderr text
    def run_main(self, responses, **constants):
        constants = {'_SANITY_MIN_MODELS': 1, **constants}
        with unittest.mock.patch.object(models, '_fetch', fake_fetch(responses)), \
             unittest.mock.patch.multiple(models, **constants), \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()) as stderr:
            models.main()
        return json.loads(stdout.getvalue()), stderr.getvalue()


    # Run the scraper's main with fake fetch responses, expecting a ValueError - returns the error message
    def run_main_error(self, responses, **constants):
        constants = {'_SANITY_MIN_MODELS': 1, **constants}
        with unittest.mock.patch.object(models, '_fetch', fake_fetch(responses)), \
             unittest.mock.patch.multiple(models, **constants), \
             unittest.mock.patch('sys.stdout', StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', StringIO()):
            with self.assertRaises(ValueError) as cm_exc:
                models.main()
        self.assertEqual(stdout.getvalue(), '')
        return str(cm_exc.exception)


    # A minimal valid scrape - a model with local, MLX, and cloud variants
    @staticmethod
    def valid_responses(*entries, **tags_pages):
        responses = {
            'https://ollama.com/library': library_page(
                library_entry('gemma', sizes=('2b', '12b', 'cloud'), capabilities=('vision',)),
                *entries
            ),
            'https://ollama.com/library/gemma/tags': tags_page('gemma', 'latest', '2b', '12b', '12b-mlx', 'cloud')
        }
        for name, page in tags_pages.items():
            responses[f'https://ollama.com/library/{name}/tags'] = page
        return responses


    def test_main(self):
        responses = {
            'https://ollama.com/library': library_page(
                library_entry('qwen', 'Qwen &amp; friends', sizes=('8b', '30b-a3b', '8b', '235b-cloud', '8b-mlx'),
                              capabilities=('tools', 'thinking', 'tools', 'speech'), pulls='12.8K',
                              updated='Jan 2, 2025 1:05 AM UTC'),
                library_entry('inkling', 'A small model: 26B parameters, 5.4B active.', capabilities=('vision', 'tools')),
                library_entry('kev', 'Kev scores answers.', capabilities=('decision',), pulls='1,234'),
                library_entry('kimi', '', capabilities=('vision',)),
                library_entry('gone', 'A removed model.', sizes=('7b',)),
                library_entry('phonon', 'Text-to-speech.'),
                library_entry('mystery', 'A mysterious model.'),
                library_entry('wizardlm', 'A 7B model.')
            ),
            'https://ollama.com/library/qwen/tags': tags_page(
                'qwen', 'latest', '8b', '30b-a3b', '8b-mlx', '8b-mlx-bf16', 'cloud', '235b-cloud'
            ),
            'https://ollama.com/library/inkling/tags': tags_page('inkling', 'latest', 'nvfp4'),
            'https://ollama.com/library/kev/tags': tags_page('kev', 'latest', '9b-q4_K_M'),
            'https://ollama.com/library/kimi/tags': tags_page('kimi', 'cloud'),
            'https://ollama.com/library/phonon/tags': tags_page('phonon', 'latest', 'en', '31m-en', '40m-en', '40m-en-bf16'),
            'https://ollama.com/library/mystery/tags': tags_page('mystery', 'latest', 'bf16'),
            'https://ollama.com/library/wizardlm/tags': tags_page('wizardlm', '7b-q4_0'),
            **registry_responses('inkling', {'model_format': 'safetensors'}),
            **registry_responses('kev', {'model_type': '9.0B'}),
            **registry_responses('mystery', {'model_type': ''})
        }
        output, stderr = self.run_main(responses, _SANITY_MAX_SKIPPED=0.5, _SANITY_MAX_DESCRIPTIONLESS=0.5)
        self.assertListEqual(output, [
            {
                'name': 'inkling',
                'description': 'A small model: 26B parameters, 5.4B active.',
                'modified': '2024-11-30',
                'downloads': 1500000,
                'capabilities': ['vision', 'tools'],
                'variants': [
                    {'id': 'inkling:latest', 'size': 'latest', 'parameters': 26000000000}
                ]
            },
            {
                'name': 'kev',
                'description': 'Kev scores answers.',
                'modified': '2024-11-30',
                'downloads': 1234,
                'capabilities': ['decision'],
                'variants': [
                    {'id': 'kev:latest', 'size': 'latest', 'parameters': 9000000000}
                ]
            },
            {
                'name': 'kimi',
                'description': 'No model description provided.',
                'modified': '2024-11-30',
                'downloads': 1500000,
                'capabilities': ['vision'],
                'variants': [
                    {'id': 'kimi:cloud', 'size': 'cloud', 'parameters': 0, 'cloud': True}
                ]
            },
            {
                'name': 'phonon',
                'description': 'Text-to-speech.',
                'modified': '2024-11-30',
                'downloads': 1500000,
                'capabilities': [],
                'variants': [
                    {'id': 'phonon:31m-en', 'size': '31m-en', 'parameters': 31000000},
                    {'id': 'phonon:40m-en', 'size': '40m-en', 'parameters': 40000000}
                ]
            },
            {
                'name': 'qwen',
                'description': 'Qwen & friends',
                'modified': '2025-01-02',
                'downloads': 12800,
                'capabilities': ['tools', 'thinking'],
                'variants': [
                    {'id': 'qwen:8b', 'size': '8b', 'parameters': 8000000000},
                    {'id': 'qwen:30b-a3b', 'size': '30b-a3b', 'parameters': 30000000000},
                    {'id': 'qwen:8b-mlx', 'size': '8b-mlx', 'parameters': 8000000000, 'mlx': True},
                    {'id': 'qwen:235b-cloud', 'size': '235b-cloud', 'parameters': 235000000000, 'cloud': True}
                ]
            }
        ])
        self.assertEqual(stderr, '''\
Info: "qwen" has unknown capability "speech"
Warning: "gone" no longer exists
Warning: "mystery" has no sizes
Warning: "wizardlm" has no sizes
''')


    def test_main_type_model_gh_pages(self):
        # On gh-pages, the type model is beside the models directory
        script_dir = os.path.dirname(os.path.abspath(models.__file__))
        gh_pages_smd = os.path.join(script_dir, '..', 'ollamaChat.smd')
        src_smd = os.path.join(script_dir, '..', '..', 'src', 'ollama_chat', 'static', 'ollamaChat.smd')
        with open(src_smd, 'r', encoding='utf-8') as fh:
            smd_text = fh.read()
        mock_open = unittest.mock.mock_open(read_data=smd_text)
        with unittest.mock.patch('os.path.isfile', return_value=True) as mock_isfile, \
             unittest.mock.patch('builtins.open', mock_open):
            output, stderr = self.run_main(self.valid_responses())
        mock_isfile.assert_called_once_with(gh_pages_smd)
        mock_open.assert_called_once_with(gh_pages_smd, 'r', encoding='utf-8')
        self.assertListEqual([model['name'] for model in output], ['gemma'])
        self.assertEqual(stderr, '')


    def test_main_invalid_downloads(self):
        responses = self.valid_responses(library_entry('bad', sizes=('7b',), pulls='lots'), bad=tags_page('bad', 'latest'))
        self.assertEqual(self.run_main_error(responses), '"bad" has invalid downloads "lots"')


    def test_main_too_few_models(self):
        self.assertEqual(
            self.run_main_error(self.valid_responses(), _SANITY_MIN_MODELS=2),
            'Suspiciously few models scraped (1 < 2)'
        )


    def test_main_too_many_skipped(self):
        responses = self.valid_responses(library_entry('gone', sizes=('7b',)))
        self.assertEqual(self.run_main_error(responses), 'Suspiciously many models skipped (1 of 2)')


    def test_main_too_many_descriptionless(self):
        responses = self.valid_responses(library_entry('bare', None, sizes=('7b',)), bare=tags_page('bare', 'latest'))
        self.assertEqual(self.run_main_error(responses), 'Suspiciously many models without descriptions (1 of 2)')


    def test_main_no_cloud(self):
        responses = self.valid_responses(gemma=tags_page('gemma', 'latest', '12b-mlx'))
        self.assertEqual(self.run_main_error(responses), 'No cloud model variants scraped')


    def test_main_no_mlx(self):
        responses = self.valid_responses(gemma=tags_page('gemma', 'latest', 'cloud'))
        self.assertEqual(self.run_main_error(responses), 'No mlx model variants scraped')


    def test_main_no_capabilities(self):
        responses = self.valid_responses()
        responses['https://ollama.com/library'] = library_page(library_entry('gemma', sizes=('2b',)))
        self.assertEqual(self.run_main_error(responses), 'No model capabilities scraped')


    def test_main_script(self):
        # Run the script - the library page fetch fails, failing the run
        with unittest.mock.patch('urllib.request.urlopen', side_effect=OSError('Network is unreachable')), \
             unittest.mock.patch('time.sleep'):
            with self.assertRaises(OSError):
                runpy.run_path(MODELS_PY, run_name='__main__')
