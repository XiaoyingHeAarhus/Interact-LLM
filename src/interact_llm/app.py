"""
Initial inspiration:https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
"""

from pathlib import Path

from llm.hf_wrapper import ChatHF
from llm.mlx_wrapper import ChatMLX
from data_models.chat import ChatHistory, ChatMessage

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

# classes for formatting
class UserMessage(Markdown):
    pass


class Response(Markdown):
    BORDER_TITLE = "Interact-LLM"


class ChatApp(App):
    """
    Texttual app for chatting with llm
    """

    AUTO_FOCUS = "INPUT"

    CSS = """
    UserMessage {
        background: $primary 10%;
        color: $text;
        margin: 1;        
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;   
        color: $text;             
        margin: 1;      
        margin-left: 8; 
        padding: 1 2 0 2;
    }
    """

    def __init__(self, model:ChatHF|ChatMLX):
        super().__init__()
        self.chat_history = ChatHistory(messages=[])
        self.model = model

    '''
    def on_mount(self) -> None:
        #self.model = ChatHF(model_id=MODEL_ID, cache_dir=CACHE_DIR, device=DEVICE)
        self.model = ChatMLX(model_id=MODEL_ID)
        self.model.load()
    '''
    def update_chat_history(self, chat_message: ChatMessage) -> None:
        """Update chat history with a single new message."""
        self.chat_history.messages.append(chat_message)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("¿Hola quieres practicar conmigo?")
        yield Input(placeholder="Escribe tu mensaje aquí")
        yield Footer()

    @on(Input.Submitted)
    async def on_input(self, user_message: Input.Submitted) -> None:
        chat_view = self.query_one("#chat-view")
        user_message.input.clear()
        await chat_view.mount(UserMessage(user_message.value))
        await chat_view.mount(response := Response())
        response.anchor()

        self.get_model_response(user_message.value, response)

    @work(thread=True)
    def get_model_response(self, user_message: str, response: Response) -> None:
        """
        Displays model response to user message, updating chat history
        """
        self.update_chat_history(ChatMessage(role="user", content=user_message))

        model_response = self.model.generate(self.chat_history)

        # replace weird <|im_end|>
        model_response.content = model_response.content.replace("<|im_end|>", "")

        # display in APP
        response_content = ""
        for chunk in model_response.content:
            response_content += chunk  # add words in a "stream-like" way
            self.call_from_thread(response.update, response_content)

        # update history again with model response
        self.update_chat_history(model_response)


def main():
    # load model with MLX if possible, default to HF instead
    try: 
        model_id = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
        print(f"[INFO]: Loading model {model_id} ... please wait")
        model = ChatMLX(model_id=model_id)
        model.load()
    except Exception as e:
        print(f"[INFO:] Failed to run using MLX. Defaulting to HuggingFace. Error: {e}")
        print(f"[INFO]: Loading model {model_id} ... please wait")
        model_id = "BSC-LT/salamandra-2b-instruct"
        cache_dir = Path(__file__).parents[3] / "models" 
        model = ChatHF(model_id=model_id, cache_dir=cache_dir)
        model.load()

    # open tui app
    app = ChatApp(model=model)
    app.run()


if __name__ == "__main__":
    main()
