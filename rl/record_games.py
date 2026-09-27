"""정책의 게임을 행동 단위로 기록한다(리플레이 페이지용 JSON).
게임을 여러 판 두고 서로 다른 결말(가장 오래 산 판, 30일 보스 사망, 일반 밤 사망 등)을 골라 저장한다.
사용: python rl/record_games.py rl/runs/cb9/agent.pt out.json [--pool 32] [--pick 5] [--seed 777]"""
import argparse
import json

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent, masked_dist

BOSS_DAYS = (10, 20, 30, 40, 50)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--pool", type=int, default=32, help="둘 게임 수 (여기서 고른다)")
    p.add_argument("--pick", type=int, default=5)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    agent.load_state_dict(ck["agent"])
    agent.eval()
    torch.manual_seed(args.seed)

    N = args.pool
    env = ts.VecEnv(N, seed=args.seed, threads=4)
    grid = np.zeros((N, C, H, W), np.float32)
    scal = np.zeros((N, S), np.float32)
    mask = np.zeros((N, A), bool)
    rew, done, days = np.zeros(N, np.float32), np.zeros(N, bool), np.zeros(N, np.float32)
    env.reset(grid, scal, mask)
    frames = [[] for _ in range(N)]
    result = [None] * N
    while any(r is None for r in result):
        with torch.no_grad():
            logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            m = torch.from_numpy(mask).to(dev)
            dist = masked_dist(logits, m)
            act = dist.sample()
            probs = dist.probs.cpu().numpy()
        act = act.cpu().numpy()
        for i in range(N):
            if result[i] is not None:
                continue
            day, hearts, swaps, phase, score = env.state(i)
            cells, turrets = env.board(i)
            top = np.argsort(-probs[i])[:3]
            frames[i].append({"d": day, "h": hearts, "s": swaps, "p": phase, "sc": score,
                              "b": [c for row in cells for c in row], "t": turrets,
                              "a": int(act[i]), "pa": round(float(probs[i][act[i]]), 3),
                              "top": [[int(x), round(float(probs[i][x]), 3)] for x in top if probs[i][x] > 0.005]})
        env.step(act, grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            i = f["env"]
            if result[i] is None:
                result[i] = {"day": f["day"], "score": f["score"], "truncated": f["truncated"]}

    games = [{"id": i, **result[i], "frames": frames[i]} for i in range(N)]
    dd = [g["day"] for g in games]
    print("풀:", sorted(dd))

    # 서로 다른 결말을 고른다: 가장 오래 산 판, 30일 보스 사망 2판, 21~29일 일반 밤 사망, 20일 보스 또는 그 전 사망
    picked = []
    def take(cands):
        for g in cands:
            if g not in picked:
                picked.append(g)
                return
    take(sorted(games, key=lambda g: -g["day"]))
    take([g for g in games if g["day"] == 30])
    take([g for g in games if 21 <= g["day"] <= 29])
    take([g for g in games if g["day"] <= 20])
    take([g for g in games if g["day"] == 30])
    for g in sorted(games, key=lambda g: abs(g["day"] - np.median(dd))):
        if len(picked) >= args.pick:
            break
        take([g])
    picked = picked[:args.pick]
    picked.sort(key=lambda g: -g["day"])
    for g in picked:
        print(f"게임 {g['id']}: {g['day']}일, 점수 {g['score']}, 행동 {len(g['frames'])}")
    with open(args.out, "w") as f:
        json.dump({"model": args.ckpt, "games": picked}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
