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

    __slots__ = ('app', 'id_', 'completed')


    def __init__(self, app, id_):
        self.app = app
        self.id_ = id_
        self.completed = False

        # Start the chat thread
        chat_thread = threading.Thread(target=OllamaChat.chat_thread_fn, args=(self,))
        chat_thread.daemon = True
        chat_thread.start()


    def stop(self):
        self.completed = True


    @staticmethod
    def chat_thread_fn(chat):
        # Create the Ollama messages from the conversation
        with chat.app.config() as config:
            conversation = next(conv for conv in config['conversations'] if conv['id'] == chat.id_)
            model = conversation['model']
            messages = []
            for exchange in conversation['exchanges']:
                messages.append({'role': 'user', 'content': exchange['user']})
                messages.append({'role': 'assistant', 'content': exchange['model']})

        # Start the chat
        stream = ollama.chat(model=model, messages=messages, stream=True)

        # Stream the chat response
        for chunk in stream:
            if chat.completed:
                break

            # Update the conversation
            with chat.app.config() as config:
                conversation = next(conv for conv in config['conversations'] if conv['id'] == chat.id_)
                exchange = conversation['exchanges'][-1]
                exchange['model'] += chunk['message']['content']

        # Write the config
        with chat.app.config(save=True):
            pass

        # Mark the chat completed
        if not chat.completed:
            chat.completed = True
        else:
            stream.close()

        # Delete the app chat entry
        del chat.app.chats[chat.id_]
