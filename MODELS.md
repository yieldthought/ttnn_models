# Model Bringup Eval Results

Prompt file: prompts/bringup_eval_long.txt
Target prompt length: 100-200 tokens (per-model tokenization varies)
Generated tokens: 100

Note: Keep the table columns padded with spaces and right-justify numeric cells so it stays aligned in terminal views. Top-1 and Top-5 are whole percents (0 d.p.).

| Model                               | Hardware |  Variant   | Top-1 | Top-5 |   TTFT | t/s/u | Seq len |
| ----------------------------------- | :------: | :--------: | ----: | ----: | -----: | ----: | ------: |
| arcee-ai/Arcee-Spark                |   n150   | functional |   86% |  100% |   94ms |  12.3 |   29952 |
| arcee-ai/Arcee-Spark                |   n300   | functional |   88% |  100% |  329ms |   4.9 |   32768 |
| arcee-ai/Arcee-Spark                |  t3000   | functional |   90% |  100% |  343ms |   4.9 |   32768 |
| arcee-ai/AFM-4.5B                   |  t3000   | functional |   98% |  100% |  181ms |   7.1 |   65536 |
| arcee-ai/AFM-4.5B                   |   n150   | functional |   98% |  100% |   72ms |  17.2 |   65536 |
| arcee-ai/AFM-4.5B                   |   n300   | functional |   97% |  100% |  283ms |   5.6 |   65536 |
| humain-ai/ALLaM-7B-Instruct-preview |   n150   | functional |   97% |  100% |   76ms |  14.9 |    4096 |
| humain-ai/ALLaM-7B-Instruct-preview |   n300   | functional |   97% |  100% |  184ms |   7.9 |     256 |
| humain-ai/ALLaM-7B-Instruct-preview |  t3000   | functional |   96% |  100% |  128ms |   9.4 |     256 |
| meta-llama/Llama-3.2-1B             |   n150   | functional |   92% |  100% |   34ms |  39.5 |  131072 |
| meta-llama/Llama-3.2-1B             |   n150   | optimized  |   79% |   99% |   26ms |  58.0 |    1024 |
| meta-llama/Llama-3.2-1B             |   n300   | functional |   90% |  100% |  610ms |   6.7 |  131072 |
| meta-llama/Llama-3.2-1B             |  t3000   | functional |   92% |  100% |  267ms |   6.6 |  131072 |
| mistralai/Mistral-7B-Instruct-v0.3  |   n150   | functional |   93% |  100% |  105ms |  16.5 |   32768 |
| mistralai/Mistral-7B-Instruct-v0.3  |   n300   | functional |   96% |  100% |  112ms |  11.1 |   32768 |
| mistralai/Mistral-7B-Instruct-v0.3  |  t3000   | functional |  100% |  100% |  110ms |  10.6 |    1024 |
| Qwen/Qwen3-0.6B                     |   n150   | functional |   99% |  100% |   52ms |  28.0 |   40960 |
| Qwen/Qwen3-0.6B                     |   n300   | functional |   99% |  100% |  329ms |   5.2 |    2048 |
| Qwen/Qwen3-0.6B                     |  t3000   | functional |   98% |  100% |  229ms |   6.2 |   40960 |
| google/gemma-3-4b-it                |   n150   | functional |   92% |  100% |   98ms |  13.9 |   40960 |
| google/gemma-3-4b-it                |   n300   | functional |   bad |   bad |  360ms |   4.6 |     256 |
| google/gemma-3-4b-it                |  t3000   | functional |   92% |  100% |  322ms |   4.8 |     256 |
| microsoft/Phi-3-mini-128k-instruct  |   n150   | functional |   92% |   99% |   80ms |  13.7 |   12288 |
| microsoft/Phi-3-mini-128k-instruct  |   n300   | functional |   90% |  100% |  168ms |   7.6 |     256 |
| microsoft/Phi-3-mini-128k-instruct  |  t3000   | functional |   bad |   bad |   bad |   bad |     256 |
| tiiuae/Falcon3-7B-Instruct          |   n150   | functional |   97% |  100% |  144ms |  13.4 |   32768 |
| tiiuae/Falcon3-7B-Instruct          |   n300   | functional |   97% |  100% |  659ms |   5.7 |    1024 |
| tiiuae/Falcon3-7B-Instruct          |  t3000   | functional |   97% |  100% |  381ms |   7.6 |    1024 |
