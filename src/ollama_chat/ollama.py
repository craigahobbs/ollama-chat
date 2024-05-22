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

    __slots__ = ('app', 'conversation_id', 'stop')


    def __init__(self, app, conversation_id):
        self.app = app
        self.conversation_id = conversation_id
        self.stop = False

        # Start the chat thread
        chat_thread = threading.Thread(target=OllamaChat.chat_thread_fn, args=(self,))
        chat_thread.daemon = True
        chat_thread.start()


    @staticmethod
    def chat_thread_fn(chat):
        try:
            # Create the Ollama messages from the conversation
            messages = []
            with chat.app.config() as config:
                conversation = config_conversation(config, chat.conversation_id)
                model = conversation['model']
                for exchange in conversation['exchanges']:
                    messages.append({'role': 'user', 'content': exchange['user']})
                    if exchange['model'] != '':
                        messages.append({'role': 'assistant', 'content': exchange['model']})

            # Start the chat
            stream = ollama.chat(model=model, messages=messages, stream=True)

            # Stream the chat response
            for chunk in stream:
                # If stopped, return immediately. The chat is deleted by the stopper.
                if chat.stop:
                    stream.close()
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


def config_conversation(config, id_):
    """
    Helper to find a conversation by ID
    """
    return next((conv for conv in config['conversations'] if conv['id'] == id_), None)
