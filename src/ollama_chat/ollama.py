# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama chat manager
"""

import threading

import ollama


class OllamaChat():
    """
    The ollama chat manager class
    """

    __slots__ = ('model', 'prompt', 'lock', 'chunks', 'completed')


    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt
        self.lock = threading.Lock()
        self.chunks = []
        self.completed = False

        # Start the chat thread
        chat_thread = threading.Thread(target=OllamaChat.chat_thread_fn, args=(self, prompt))
        chat_thread.daemon = True
        chat_thread.start()


    def stop(self):
        with self.lock:
            self.completed = True


    def get_response(self):
        with self.lock:
            return (''.join(self.chunks), self.completed)


    def add_chunk(self, chunk):
        with self.lock:
            self.chunks.append(chunk)


    @staticmethod
    def chat_thread_fn(chat, prompt):
        # Start the chat
        stream = ollama.chat(
            model=chat.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            stream=True
        )

        # Stream the chat response
        for chunk in stream:
            chat.add_chunk(chunk['message']['content'])
            if chat.completed:
                break

        # Mke sure the stream is closed
        stream.close()

        # Mark the chat completed
        chat.completed = True
