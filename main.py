from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
import importlib.metadata

lg_version = importlib.metadata.version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

import os
print(os.getenv("GOOGLE_API_KEY"))

def main():
    # Test openai
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from GoogleGenerativeAI : {response}")

    # Test anthropic
    # llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    # response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    # print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup complete!")

if __name__ == "__main__":
    main()
