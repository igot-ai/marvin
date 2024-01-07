from logging import Logger
from typing import Callable, Union

from litellm import acompletion

from marvin.engine.language_models import OpenAIFunction
from marvin.engine.language_models.open_ai import OpenAIStreamHandler, OpenAIChatLLM
from marvin.utilities.messages import Message, Role


class LiteLLM(OpenAIChatLLM):

    def format_messages(self, messages: list[Message]) -> Union[str, dict, list[Union[str, dict]]]:
        return super().format_messages(messages)

    async def run(self, messages: list[Message], functions: list[OpenAIFunction] = None, *, logger: Logger = None,
                  stream_handler: Callable[[Message], None] = False, **kwargs) -> Message:
        prompts = self.format_messages(messages)

        response = await acompletion(
            model=self.model,
            messages=prompts,
            stream=True if stream_handler else False,
            **kwargs
        )
        if stream_handler:
            handler = OpenAIStreamHandler(callback=stream_handler)
            msg = await handler.handle_streaming_response(response)
            role = msg.role

            if role == Role.ASSISTANT and isinstance(
                    msg.data.tool_calls, dict
            ):
                role = Role.FUNCTION_REQUEST

            msg = Message(
                role=role,
                content=msg.content,
                data=msg.data,
                llm_response=msg.llm_response,
            )
        else:
            llm_response = response.model_dump()
            msg = llm_response["choices"][0]["message"].copy()
            role = msg.pop("role").upper()
            if role == "ASSISTANT" and isinstance(msg.get("function_call"), dict):
                role = Role.FUNCTION_REQUEST
            msg = Message(
                role=role,
                content=msg.pop("content", None),
                data=msg,
                llm_response=llm_response,
            )

        return msg
