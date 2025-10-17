from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


resp = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "What is the capital of Australia?"}]
)

print(resp.output_text)
