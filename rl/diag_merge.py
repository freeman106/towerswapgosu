"""진단: 연습 시작 상태(난이도 1)에서 정책이 상자 합성 행동과 상자 개봉에 주는 확률.
사용: python rl/diag_merge.py rl/runs/cb3/agent.pt [--level 1] [--states 1024]"""
import argparse

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--level", type=int, default=1)
p.add_argument("--states", type=int, default=1024)
args = p.parse_args()
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location=dev)
agent = Agent(*(ck["args"][k] for k in ("channels", "blocks", "hidden"))).to(dev)
agent.load_state_dict(ck["agent"])

N = args.states
env = ts.VecEnv(N, seed=77)
env.build_start_pool(N, args.level, seed=77)
env.set_start_frac(1.0)
grid, scal, mask = np.zeros((N, C, H, W), np.float32), np.zeros((N, S), np.float32), np.zeros((N, A), bool)
env.reset(grid, scal, mask)
merge = np.zeros((N, A), bool)
env.merge_actions(merge)
# 상자 탭: 칸 선택 행동 중 상자가 있는 칸 (격자 채널: 종류 원-핫의 상자 = 12)
chest_cell = grid[:, 12].reshape(N, -1) > 0.5  # [N, 48] (y*6 + x-1)
tap_chest = np.zeros((N, A), bool)
tap_chest[:, 168:216] = chest_cell & mask[:, 168:216]
with torch.no_grad():
    logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
    prob = torch.softmax(logits.masked_fill(~torch.from_numpy(mask).to(dev), -1e9), 1).cpu().numpy()
has = merge.any(1)
n_valid = mask.sum(1)
print(f"상태 {N}개 중 합성 행동이 있는 상태 {has.mean():.0%}, 합성 행동 수 평균 {merge[has].sum(1).mean():.2f}, 유효 행동 수 평균 {n_valid.mean():.1f}")
print(f"정책 확률 — 합성 행동 합 {prob[has][merge[has]].sum() / has.sum():.3f} (균등이면 {(merge[has].sum(1) / n_valid[has]).mean():.3f}), "
      f"상자 개봉 합 {(prob * tap_chest).sum(1)[has].mean():.3f}, 최빈 행동이 합성 {merge[has, prob[has].argmax(1)].mean():.0%}")
ent = -(prob * np.log(prob + 1e-12)).sum(1)
print(f"정책 엔트로피 평균 {ent[has].mean():.3f}")
