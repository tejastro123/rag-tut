import tiktoken

class TokenBudget:

def __init__(self, max_tokens=4000):

    self.max_tokens = max_tokens

    self.encoding = tiktoken.get_encoding(
        "cl100k_base"
    )

def count_tokens(self, text: str):

    return len(
        self.encoding.encode(text)
    )

def validate(self, text: str):

    tokens = self.count_tokens(text)

    if tokens > self.max_tokens:
        raise ValueError(
            f"Token limit exceeded: {tokens}"
        )

    return tokens
