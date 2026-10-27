# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic background load for the decode-regulation sidecar smoke test.

Keeps --concurrency long-context requests decoding without a priority (so they
are regulated) until --duration-s elapses. Each prompt is unique, so the prefix
cache cannot shortcut prefill. Writes one JSON line per request to --out.
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
]  # fmt: skip


def prompt(rng: random.Random, tokens: int) -> str:
    # About one token per word for these words; a numeric tag keeps prefixes unique.
    return f"[{rng.random():.12f}] " + " ".join(rng.choices(WORDS, k=tokens))


async def worker(i, args, session, deadline, out):
    rng = random.Random(args.seed * 1000 + i)
    while time.time() < deadline:
        body = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt(rng, args.prompt_tokens)
                    + "\nRepeat the words above, one per line.",
                }
            ],
            "max_tokens": args.max_tokens,
            "min_tokens": args.max_tokens,
            "ignore_eos": True,
            "stream": False,
        }
        t0 = time.time()
        rec = {"worker": i, "t_start": t0}
        try:
            async with session.post(f"{args.base_url}/v1/chat/completions", json=body) as r:
                data = await r.json(content_type=None)
                rec["status"] = r.status
                rec["completion_tokens"] = (data.get("usage") or {}).get(
                    "completion_tokens"
                )
                if r.status != 200:
                    rec["error"] = str(data)[:300]
        except Exception as e:  # noqa: BLE001 - record and keep the load going
            rec["status"] = -1
            rec["error"] = repr(e)[:300]
            await asyncio.sleep(1)
        rec["t_done"] = time.time()
        out.write(json.dumps(rec) + "\n")
        out.flush()


async def main(args):
    deadline = time.time() + args.duration_s
    timeout = aiohttp.ClientTimeout(total=args.duration_s + 600)
    conn = aiohttp.TCPConnector(limit=args.concurrency + 4)
    with open(args.out, "w") as out:
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            tasks = []
            for i in range(args.concurrency):
                tasks.append(asyncio.create_task(worker(i, args, session, deadline, out)))
                await asyncio.sleep(args.ramp_s / args.concurrency)
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--prompt-tokens", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--duration-s", type=float, default=480)
    p.add_argument("--ramp-s", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(main(p.parse_args()))
