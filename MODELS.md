# Model Bringup Eval Results

Prompt file: prompts/bringup_eval_long.txt
Target prompt length: 100-200 tokens (per-model tokenization varies)
Generated tokens: 100

Command template:
python eval.py <model_path> --model <hf_id> --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100

Commands used:
python eval.py models/meta-llama/Llama-3.2-1B/n150/functional/model.py --model meta-llama/Llama-3.2-1B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
TT_VISIBLE_DEVICES=0,2 python eval.py models/meta-llama/Llama-3.2-1B/n300/functional/model.py --model meta-llama/Llama-3.2-1B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/mistralai/Mistral-7B-Instruct-v0.3/n150/functional/model.py --model mistralai/Mistral-7B-Instruct-v0.3 --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/Qwen/Qwen3-0.6B/n150/functional/model.py --model Qwen/Qwen3-0.6B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
TT_VISIBLE_DEVICES=0,2 TT_METAL_CACHE=/tmp/tt-metal-cache TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 python eval.py models/Qwen/Qwen3-0.6B/n300/functional/model.py --model Qwen/Qwen3-0.6B --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/google/gemma-3-4b-it/n150/functional/model.py --model google/gemma-3-4b-it --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
TT_VISIBLE_DEVICES=0,2 TT_METAL_CACHE=/tmp/tt-metal-cache TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root TT_METAL_INSPECTOR_LOG_PATH=/tmp/tt-metal-inspector TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 python eval.py models/google/gemma-3-4b-it/n300/functional/model.py --model google/gemma-3-4b-it --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/microsoft/Phi-3-mini-128k-instruct/n150/functional/model.py --model microsoft/Phi-3-mini-128k-instruct --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/tiiuae/Falcon3-7B-Instruct/n150/functional/model.py --model tiiuae/Falcon3-7B-Instruct --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100
python eval.py models/humain-ai/ALLaM-7B-Instruct-preview/n150/functional/model.py --model humain-ai/ALLaM-7B-Instruct-preview --prompt_file prompts/bringup_eval_long.txt --max_new_tokens 100

| Model | Model Path | HF ID | Prompt Tokens | Max New Tokens | Top-1 | Top-5 | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALLaM-7B-Instruct-preview | models/humain-ai/ALLaM-7B-Instruct-preview/n150/functional/model.py | humain-ai/ALLaM-7B-Instruct-preview | 146 | 100 | 95.00% | 100.00% | pass | MAX_CACHE_SEQ_LEN 256, sharded fill_cache |
| Llama-3.2-1B | models/meta-llama/Llama-3.2-1B/n150/functional/model.py | meta-llama/Llama-3.2-1B | 141 | 100 | 93.00% | 100.00% | pass | - |
| Llama-3.2-1B (n300) | models/meta-llama/Llama-3.2-1B/n300/functional/model.py | meta-llama/Llama-3.2-1B | 141 | 100 | 91.00% | 100.00% | pass | 1x2 mesh |
| Mistral-7B-Instruct-v0.3 | models/mistralai/Mistral-7B-Instruct-v0.3/n150/functional/model.py | mistralai/Mistral-7B-Instruct-v0.3 | 155 | 100 | 93.00% | 100.00% | pass | - |
| Qwen3-0.6B | models/Qwen/Qwen3-0.6B/n150/functional/model.py | Qwen/Qwen3-0.6B | 140 | 100 | 99.00% | 100.00% | pass | - |
| Qwen3-0.6B (n300) | models/Qwen/Qwen3-0.6B/n300/functional/model.py | Qwen/Qwen3-0.6B | 140 | 100 | 99.00% | 100.00% | pass | 1x2 mesh |
| Gemma-3-4b-it | models/google/gemma-3-4b-it/n150/functional/model.py | google/gemma-3-4b-it | 139 | 100 | 92.00% | 100.00% | pass | MAX_CACHE_SEQ_LEN 256 |
| Gemma-3-4b-it (n300) | models/google/gemma-3-4b-it/n300/functional/model.py | google/gemma-3-4b-it | 139 | 100 | 90.00% | 100.00% | pass | 1x2 mesh, MAX_CACHE_SEQ_LEN 256 |
| Phi-3-mini-128k-instruct | models/microsoft/Phi-3-mini-128k-instruct/n150/functional/model.py | microsoft/Phi-3-mini-128k-instruct | 155 | 100 | 90.00% | 99.00% | pass | MAX_CACHE_SEQ_LEN 256, sharded fill_cache |
| Falcon3-7B-Instruct | models/tiiuae/Falcon3-7B-Instruct/n150/functional/model.py | tiiuae/Falcon3-7B-Instruct | 137 | 100 | 97.00% | 100.00% | pass | MAX_CACHE_SEQ_LEN 1024 |
