"""리플레이용 짝 기록: 같은 게임을 기본 정책과 탐색 에이전트가 각각 둔 기록(첫 탐색 교체 전까지 동일)을 저장한다.
결말이 서로 다른 판을 골라 JSON으로 쓴다(record_games.py와 같은 프레임 형식 + 탐색 후보).
사용: python rl/record_search_games.py rl/runs/cb9/agent.pt out.json [--pool 32] [--pick 5] [--seed 12345]"""
import argparse
import json

import numpy as np
import torch

from search_eval import Policy, play


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--pool", type=int, default=32)
    p.add_argument("--pick", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.pool
    fb, fs = [[] for _ in range(n)], [[] for _ in range(n)]
    base = play(pol, n, args.seed, False, 0, 0, 0, None, frames=fb)
    stats = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
    srch = play(pol, n, args.seed, True, 3, 2, 8, stats, 2.0, frames=fs)
    diff = srch - base
    print("기본:", base.tolist())
    print("탐색:", srch.tolist())
    # 고르기: 탐색 이득이 가장 큰 판, 기본이 30일 보스에서 죽고 탐색이 넘긴 판, 전형적인 개선(중앙값 근처), 탐색이 더 나빴던 판, 둘 다 비슷한 판
    order = []
    def take(c):
        for i in c:
            if i not in order:
                order.append(int(i))
                return
    take(np.argsort(-diff))
    take([i for i in np.argsort(-srch) if base[i] == 30 and srch[i] > 30])
    med = np.median(diff[diff > 0]) if (diff > 0).any() else 0
    take(sorted(np.nonzero(diff > 0)[0], key=lambda i: abs(diff[i] - med)))
    take(np.argsort(diff))
    take(sorted(range(n), key=lambda i: abs(diff[i])))
    for i in np.argsort(-srch):
        if len(order) >= args.pick:
            break
        take([i])
    games = []
    for i in order[:args.pick]:
        first_div = next((k for k, (a, b) in enumerate(zip(fb[i], fs[i])) if a["a"] != b["a"]), None)
        games.append({"id": i, "base": {"day": int(base[i]), "frames": fb[i]}, "search": {"day": int(srch[i]), "frames": fs[i]},
                      "diverge": first_div})
        print(f"게임 {i}: 기본 {base[i]}일 → 탐색 {srch[i]}일 · 첫 갈림 {first_div}번째 행동")
    with open(args.out, "w") as f:
        json.dump({"model": args.ckpt, "games": games}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
