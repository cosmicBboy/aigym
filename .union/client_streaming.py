import json
from typing import TypedDict

import httpx

MAX_RETRIES = 10


class ChatCompletionMessageChunk(TypedDict):
    role: str
    content: str


class ChatCompletionChoiceChunk(TypedDict):
    index: int
    delta: ChatCompletionMessageChunk
    finish_reason: str


class ChatCompletionChunk(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoiceChunk]


def run(url: str, message: str):
    print("Starting stream")
    for _ in range(MAX_RETRIES):
        try:
            with httpx.stream(
                "POST",
                f"{url}/v1/chat/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "messages": [{"role": "user", "content": message}],
                    "stream": True,
                    "logprobs": True,
                },
                timeout=180,
            ) as response:
                for chunk in response.iter_lines():
                    if chunk == "":
                        continue
                    json_chunk = json.loads(chunk.replace("data: ", ""))
                    if json_chunk["choices"][0]["finish_reason"] == "stop":
                        break
                    text = json_chunk["choices"][0]["delta"]["content"]
                    yield text
            return
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    default_url = "https://shy-violet-c90de.apps.serverless-1.us-east-2.s.union.ai"
    parser.add_argument("--url", type=str, default=default_url)
    parser.add_argument("--message", type=str, default="Are you online?")
    args = parser.parse_args()

    final_text = ""
    for text in run(args.url, args.message):
        final_text += text
        print(text, end="", flush=True)
