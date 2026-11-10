# Community benchmark on Radeon (TM) RX 470 Graphics

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Radeon (TM) RX 470 Graphics, 8 GiB VRAM
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 31.0.21923.11000; CUDA: not read
- Strata: project 0.1.47, commit 1158e0bf; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv q4lm8v --vision
```

Environment of the config: `STRATA_KV_WINDOW=1`, `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 35.9 (32.2-42.6) | 0.84 (0.71-1.01) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 485.7 (477.3-486.3) | not measured | 8.31 (8.30-8.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 543.4 (541.6-556.1) | not measured | 29.60 (28.95-29.70) |
| agent session, follow-up turns | 14736 (14652-14740) at the last turn | see below | up to 64 | 3 | see below | 36.8 (30.7-48.4) | 7.99 (6.75-8.58) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.5); earlier-turn tokens read again per session: 74 (72-74) (earlier turns that the cache did not hold; the 13858 (13857-13858) new tokens of the files are read once); first turn (nothing cached) first token: 4.45 s (4.36-4.50).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 141.0 | 4.355 | 21.7 |
| 1 | 4095 | 624 | 3471 | 421.9 | 8.271 | 32.5 |
| 2 | 5628 | 4088 | 1540 | 229.1 | 6.774 | 30.8 |
| 3 | 9188 | 5693 | 3495 | 411.4 | 8.549 | 39.8 |
| 4 | 11960 | 9251 | 2709 | 343.2 | 7.952 | 48.4 |
| 5 | 14740 | 12023 | 2717 | 344.3 | 7.957 | 30.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 315 s.
