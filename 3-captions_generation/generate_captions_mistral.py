"""
⚠️ vllm is too buggy. I was not able to pre-download the model before submitting this script to a computation node, which doesn't have internet access.
Mistral-Small-3.1-24B-Instruct-2503 is very powerful, but for the moment this script is abandoned.
"""

from pathlib import Path
from vllm import LLM
from vllm.sampling_params import SamplingParams

ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT_DIR / ".huggingface"

SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."

user_prompt = "Give me 5 non-formal ways to say 'See you later' in French."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]

model_name = CACHE_PATH / "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
model_path = (
    CACHE_PATH
    / "models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/73ce7c62b904fa83d7cb018e44c3bc06feed4d81"
)
llm = LLM(model=str(model_path), tokenizer_mode="mistral", download_dir=str(CACHE_PATH))

sampling_params = SamplingParams(max_tokens=64, temperature=0.65)
outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
