# Model Bringup Eval Results

Prompt file: prompts/bringup_eval_long.txt
Target prompt length: 100-200 tokens (per-model tokenization varies)
Generated tokens: 100

Note: Keep the table columns padded with spaces and right-justify numeric cells so it stays aligned in terminal views. Top-1 and Top-5 are whole percents (0 d.p.).

| Model                               | Hardware | Variant    | Top-1 | Top-5 | TTFT | t/s/u | Seq len |
| ----------------------------------- | -------- | ---------- | ----- | ----- | ---- | ----- | ------- |
| arcee-ai/Arcee-Spark                | n150     | functional |   90% |  100% |      |       |   29952 |
| arcee-ai/AFM-4.5B                   | n150     | functional |   98% |  100% | 72ms |  17.2 |   65536 |
| humain-ai/ALLaM-7B-Instruct-preview | n150     | functional |   95% |  100% |      |       |    4096 |
| humain-ai/ALLaM-7B-Instruct-preview | n300     | functional |   98% |  100% |      |       |     256 |
| humain-ai/ALLaM-7B-Instruct-preview | t3000    | functional |   97% |  100% |      |       |     256 |
| meta-llama/Llama-3.2-1B             | n150     | functional |   92% |  100% |      |       |  131072 |
| meta-llama/Llama-3.2-1B             | n300     | functional |   91% |  100% |      |       |         |
| meta-llama/Llama-3.2-1B             | t3000    | functional |   87% |  100% |      |       |         |
| mistralai/Mistral-7B-Instruct-v0.3  | n150     | functional |   93% |  100% |      |       |   32768 |
| mistralai/Mistral-7B-Instruct-v0.3  | n300     | functional |   96% |  100% |      |       |    1024 |
| mistralai/Mistral-7B-Instruct-v0.3  | t3000    | functional |   97% |  100% |      |       |    1024 |
| Qwen/Qwen3-0.6B                     | n150     | functional |   99% |  100% |      |       |   40960 |
| Qwen/Qwen3-0.6B                     | n300     | functional |   99% |  100% |      |       |         |
| Qwen/Qwen3-0.6B                     | t3000    | functional |   99% |  100% |      |       |         |
| google/gemma-3-4b-it                | n150     | functional |   92% |  100% |      |       |   40960 |
| google/gemma-3-4b-it                | n300     | functional |   90% |  100% |      |       |     256 |
| google/gemma-3-4b-it                | t3000    | functional |   91% |  100% |      |       |     256 |
| microsoft/Phi-3-mini-128k-instruct  | n150     | functional |   90% |   99% |      |       |   12288 |
| microsoft/Phi-3-mini-128k-instruct  | n300     | functional |   92% |   99% |      |       |     256 |
| microsoft/Phi-3-mini-128k-instruct  | t3000    | functional |   92% |  100% |      |       |     256 |
| tiiuae/Falcon3-7B-Instruct          | n150     | functional |   97% |  100% |      |       |   32768 |
| tiiuae/Falcon3-7B-Instruct          | n300     | functional |   97% |  100% |      |       |    1024 |
| tiiuae/Falcon3-7B-Instruct          | t3000    | functional |   98% |  100% |      |       |    1024 |
