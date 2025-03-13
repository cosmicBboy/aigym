import httpx


def main(url: str, message: str):
    completion = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "messages": [{"role": "user", "content": message}],
        },
        timeout=180,
    )
    print(completion.text)


if __name__ == "__main__":
    main(
        "https://shy-violet-c90de.apps.serverless-1.us-east-2.s.union.ai",
        "Are you online?",
    )
