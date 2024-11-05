import asyncio
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import aiohttp
from dataclasses import dataclass
from typing import AsyncGenerator, List, Tuple
from tqdm.asyncio import tqdm

import aiohttp.client_exceptions

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    api_url: str
    prompt: str
    prompt_len: int
    output_len: int
    model: str


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0  # Time to first token
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""
    
    
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


async def async_request_chat(
    request_func_input: RequestFuncInput,
    pbar: tqdm
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "/chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'v1/chat/completions'."
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": True,
            "stop": [],
            "stream_options": {      
                "include_usage": True
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        chunk_list = []
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk_list.append(chunk)
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)
                            if len(data["choices"]) > 0:
                                first_choice = data["choices"][0]
                                delta = first_choice["delta"]
                                if delta.get("content", None):
                                    # First token
                                    if ttft == 0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft
                                    generated_text += delta["content"]
                            
                            usage = data.get("usage", None)
                            if usage:   # the first chunk is none
                                output.output_len = usage.get("completion_tokens", 0)

                    output.generated_text = generated_text
                    if generated_text == "" and output.output_len == 0:
                        output.success = False
                        output.error = f"error_msg: Didn't generate any tokens. response.status: {response.status}. chunk_list: {chunk_list}"
                        print(f"async_request_chat error {output.error}")
                    else:
                        output.success = True
                        output.latency = latency

                else:
                    print(response.status, response.reason, response.text)
                    output.error = f"{response.status} {response.reason} {response.text}"
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(f"async_request_chat Exception {output.error}")

    pbar.update(1)
    return output


def load_dataset(dataset_path: str, num_requests: int) -> List[Tuple[str, int, int]]:
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    filtered_dataset = [(data["prompt"], data["prompt_len"], data["output_len"]) for data in dataset]
    
    assert len(filtered_dataset) >= num_requests, "Dataset does not contain enough entries."
    
    return filtered_dataset[:num_requests]


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    for request in input_requests:
        yield request

        if request_rate != float("inf"):
            await asyncio.sleep(np.random.exponential(1.0 / request_rate))
        
        
async def benchmark(
    api_url: str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    pbar: tqdm
):
    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model = model_id,
            prompt = prompt,
            api_url = api_url,
            prompt_len = prompt_len,
            output_len = output_len
        )
        tasks.append(asyncio.create_task(async_request_chat(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    # calculate metrics
    completed = 0
    total_input = 0
    actual_output_lens = []
    tpots = []
    ttfts = []
    for i in range(len(outputs)):
        total_input += outputs[i].prompt_len
        if outputs[i].success:
            completed += 1
            actual_output_lens.append(outputs[i].output_len)
            ttfts.append(outputs[i].ttft)
            if outputs[i].output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (outputs[i].output_len - 1))
        else:
            actual_output_lens.append(0)
    
    # print
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Successful requests:", completed))
    print("{:<40} {:<10}".format("Total input tokens:", total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", sum(actual_output_lens)))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", completed / benchmark_duration))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", sum(actual_output_lens) / benchmark_duration))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Avg TTFT (ms):", np.mean(ttfts or 0) * 1000))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", np.percentile(ttfts or 0, 90) * 1000))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Avg TPOT (ms):", np.mean(tpots or 0) * 1000))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", np.percentile(tpots or 0, 90) * 1000))
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the model API.")
    parser.add_argument("--dataset_path", type=str, default="./dataset.json", help="Path to the dataset JSON file.")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to load from the dataset.")
    parser.add_argument("--request_rate", type=float, default=4.0, help="Rate of requests per second.")
    parser.add_argument("--model", type=str, required=True, help="Model id.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/chat/completions", help="API URL.")

    args = parser.parse_args()
    
    input_requests = load_dataset(
        dataset_path = args.dataset_path,
        num_requests = args.num_prompts,
    )
    
    pbar = tqdm(total=len(input_requests))
    
    asyncio.run(benchmark(
        api_url=args.url,
        model_id=args.model,
        input_requests=input_requests,
        request_rate=args.request_rate,
        pbar = pbar
    ))
    
    pbar.close()
        