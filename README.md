## Installation

```bash
pip install -r requirements.txt
```

## Test
```bash
export OPENAI_API_KEY=my_key
python test_perf.py \
    --dataset_path ./dataset.json \
    --model /share/datasets/tmp_share/zhaoyingzhuo/qwen1.5-72B-chat-llmcompressor-fp8/ \
    --url http://0.0.0.0:8000/v1/chat/completions \
    --num_prompts 1000 \
    --request_rate inf
```
