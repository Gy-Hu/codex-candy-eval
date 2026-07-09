#!/usr/bin/env python3
"""不经任何 agent CLI，直连 OpenAI 兼容 API 测试糖果问题（题目与判分规则同 codex_candy_eval.py）。

适合评测 codex/claude/opencode 未接入的模型（如中转站上的 grok、glm 等）：

    YUNWU_API_KEY=sk-xxx python3 api_candy_eval.py -m grok-4.5 -n 5

API key 依次从 YUNWU_API_KEY、OPENAI_API_KEY 环境变量读取；都未设置时尝试
opencode 的 auth store（~/.local/share/opencode/auth.json 的 yunwu.key）。
--base-url 可指向任意 OpenAI 兼容端点，默认 https://yunwu.ai/v1。
注意：直连 API 没有 agent 系统提示词/工具包装，结果与 codex 版不严格可比。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """不使用任何外部工具回答以下问题：

在一个黑色的袋子里放有三种口味的糖果，每种糖果有两种不同的形状（圆形和五角星形，不同的形状靠手感可以分辨）。现已知不同口味的糖和不同形状的数量统计如下表。参赛者需要在活动前决定摸出的糖果数目，那么，最少取出多少个糖果才能保证手中同时拥有不同形状的苹果味和桃子味的糖？（同时手中有圆形苹果味匹配五角星桃子味糖果，或者有圆形桃子味匹配五角星苹果味糖果都满足要求）

        苹果味  桃子味  西瓜味
圆形       7      9      8
五角星形   7      6      4
"""

# 正确答案为 21：只要回答中出现独立的 "21"（前后非数字）即判为正确。
ANSWER_PATTERN = re.compile(r"(?<!\d)21(?!\d)")


def resolve_api_key() -> str:
    for env in ("YUNWU_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    auth = Path.home() / ".local/share/opencode/auth.json"
    if auth.is_file():
        key = json.loads(auth.read_text()).get("yunwu", {}).get("key")
        if key:
            return key
    sys.exit("未找到 API key：请设置 YUNWU_API_KEY 或 OPENAI_API_KEY 环境变量。")


def run_one(base_url: str, key: str, model: str, index: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 32768,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        elapsed = time.perf_counter() - start
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        rea = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return {"run": index, "ok": bool(ANSWER_PATTERN.search(text)),
                "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                "reason": rea, "time": round(elapsed, 1),
                "tail": text.replace("\n", " ")[-120:]}
    except Exception as exc:
        return {"run": index, "ok": None, "error": str(exc)[:200],
                "time": round(time.perf_counter() - start, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-n", "--tests", type=int, default=1)
    parser.add_argument("-j", "--jobs", type=int, default=5,
                        help="并发请求数（默认 5）")
    parser.add_argument("--base-url", default="https://yunwu.ai/v1",
                        help="OpenAI 兼容端点（默认 yunwu）")
    args = parser.parse_args()
    key = resolve_api_key()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(
            lambda i: run_one(args.base_url, key, args.model, i),
            range(1, args.tests + 1)))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    graded = [r["ok"] for r in results if r["ok"] is not None]
    if graded:
        print(f"\n{args.model}: {sum(graded)}/{len(graded)} correct, "
              f"accuracy={sum(graded)/len(graded)*100:.0f}%")
    else:
        print(f"\n{args.model}: 全部请求失败，未能判分")


if __name__ == "__main__":
    main()
