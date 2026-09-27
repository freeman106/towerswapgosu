"""체크포인트 평가: 학생 정책만으로 게임을 끝까지 둔다.
사용: python rl/eval.py rl/runs/bc1/agent.pt [--games 1000] [--mode argmax|sample]"""
import argparse

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent, episode_summary, format_summary, masked_dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--mode", default="argmax", choices=["argmax", "sample"])
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    agent.load_state_dict(ck["agent"])
    agent.eval()

    N = args.envs
    env = ts.VecEnv(N, seed=args.seed)
    grid = np.zeros((N, C, H, W), np.float32)
    scal = np.zeros((N, S), np.float32)
    mask = np.zeros((N, A), bool)
    rew, done, days = np.zeros(N, np.float32), np.zeros(N, bool), np.zeros(N, np.float32)
    env.reset(grid, scal, mask)
    # 환경마다 정확히 k게임씩 센다(먼저 끝난 게임만 모으면 짧은 게임 쪽으로 치우친다)
    k = -(-args.games // N)
    per = [[] for _ in range(N)]
    while min(len(x) for x in per) < k:
        with torch.no_grad():
            logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            m = torch.from_numpy(mask).to(dev)
            if args.mode == "argmax":
                act = logits.masked_fill(~m, -1e9).argmax(1)
            else:
                act = masked_dist(logits, m).sample()
        env.step(act.cpu().numpy(), grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            if len(per[f["env"]]) < k:
                per[f["env"]].append(f)
    fin = [f for x in per for f in x]
    dy = np.array([f["day"] for f in fin])
    s = episode_summary(fin)
    print(f"{args.ckpt} ({args.mode}): {format_summary(s)} | day 중앙 {int(np.median(dy))} · 보스 통과 평균 {s['bosses']:.2f} · "
          f"동/은/금 도달 {s['chest_ge2']:.0%}/{s['chest_ge3']:.0%}/{s['chest_ge4']:.0%} | "
          f"20일 보스 통과 {np.mean(dy > 20):.1%} · 30/40/50일 도달 {np.mean(dy >= 30):.1%}/{np.mean(dy >= 40):.1%}/{np.mean(dy >= 50):.1%} · "
          f"30일 보스 통과 {np.mean(dy > 30):.1%}")


if __name__ == "__main__":
    main()
