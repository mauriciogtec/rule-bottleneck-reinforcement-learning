import collections
import os
import requests
import json
from typing import Dict, List, Literal, NamedTuple, Optional, Any


Reponse = collections.namedtuple("Response", ["content"])


class HUITMistralModel:
    """
    Custom chat model for a HUIT AWS Bendrock endpoint.
    **Only for internal use at Harvard.
    """

    def __init__(
        self,
        model: str = "mistral.mistral-large-2407-v1:0",
    ):
        metadata = {}
        metadata["endpoint_url"] = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1"
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model
        self.metadata = metadata

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Convert from LangChain messages -> the custom format
        aws_style_messages = []
        for msg in messages:
            aws_style_messages.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": str(msg["content"])}],
                }
            )

        # 2. Construct the payload
        payload = json.dumps(
            {
                "modelId": self.model,
                "contentType": "application/json",
                "accept": "application/json",
                "body": {
                    "messages": aws_style_messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
            }
        )

        # 3. Send the request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.metadata["api_key"],
        }
        response = requests.post(
            self.metadata["endpoint_url"],
            headers=headers,
            data=payload,
            # timeout=60,
        )
        response.raise_for_status()
        result_json = response.json()

        # 4. Parse the response
        #    Adjust based on how your custom endpoint returns the data
        #    For example, maybe the response has a field "generated_text":
        # content = result_json.get("generated_text", "")
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)

    def get_num_tokens(self, text: str) -> int:
        # Optionally implement a real token counter
        return len(text.split())


if __name__ == "__main__":
    # llm = HUITChatModel("mistral.mistral-large-2407-v1:0")
    llm = HUITMistralModel("meta.llama3-1-70b-instruct-v1:0")
    messages = [
        {"role": "system", "content": "Be angry! Answer angry to every message."},
        {"role": "user", "content": "What's your mood?"},
    ]

    result = llm.invoke(messages, max_tokens=10)
    print(result.content)
