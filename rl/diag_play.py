"""플레이 진단: 스왑(드래그)을 어디에 쓰는지(매치 교환·버리기·성 보수 등, 버린 타일 종류)와
타워 종류·등급별 타워 1개가 하룻밤에 드래곤에게 주는 실제 피해를 잰다.
사용: python rl/diag_play.py rl/runs/long1/agent_40M.pt [--games 512]"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W
from search_eval import Policy, act_cat

GROUP = {"l": "나무", "s": "돌", "i": "철", "g": "보물", "d": "얼음", "t": "화살탑", "b": "발리스타", "c": "대포", "w": "얼음벽", "h": "상자"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.games
    env = ts.VecEnv(n, seed=args.seed)
    grid, scal, mask = np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    env.reset(grid, scal, mask)
    alive = np.ones(n, bool)
    uses = defaultdict(Counter)   # 날짜 구간 → 드래그 분류 수
    tossed = defaultdict(Counter)  # 날짜 구간 → 버린 타일
    fins = []
    step = 0
    bucket = lambda d: "1~9" if d < 10 else "10~19" if d < 20 else "20~29" if d < 30 else "30+"
    while alive.any():
        logits, _ = pol(grid, scal, mask)
        act = (logits + np.random.default_rng([args.seed, step]).gumbel(size=(n, A))).argmax(1)
        for i in np.nonzero(alive)[0]:
            day, hearts, swaps, phase, _ = env.state(int(i))
            a = int(act[i])
            if phase != 0 or a >= 168:
                continue
            cells, _ = env.board(int(i))
            cat = act_cat(cells, phase, a)
            b = bucket(day)
            uses[b][cat] += 1
            if cat == "버리기":
                c = a // 4
                src = cells[c // 6][c % 6]
                tossed[b][GROUP.get(src[0], src[0]) + (src[1] if src[1] not in "0" else "")] += 1
        env.step(act, grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            if alive[f["env"]]:
                alive[f["env"]] = False
                fins.append(f)
        step += 1
    dd = np.array([f["day"] for f in fins])
    print(f"{args.ckpt} 정책 혼자 {len(fins)}판: day {dd.mean():.2f}")
    print("\n[드래그(스왑) 사용처] 날짜 구간별 비율")
    cats = [c for c, _ in sum(uses.values(), Counter()).most_common(8)]
    print("  구간    | " + " | ".join(f"{c}" for c in cats))
    for b in ("1~9", "10~19", "20~29", "30+"):
        tot = sum(uses[b].values())
        if tot:
            print(f"  {b:6s} | " + " | ".join(f"{uses[b][c] / tot:5.1%}" for c in cats) + f"   (드래그 {tot})")
    print("\n[버린 타일] 날짜 구간별 상위")
    for b in ("1~9", "10~19", "20~29", "30+"):
        tot = sum(tossed[b].values())
        if tot:
            print(f"  {b:6s}: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in tossed[b].most_common(6)))
    print("\n[타워 1개가 하룻밤에 주는 피해] 종류 × 등급 (타워-밤 수)")
    dmg = np.sum([f["dmg"] for f in fins], 0)
    tn = np.sum([f["tower_nights"] for f in fins], 0)
    for k, name in enumerate(("화살탑", "발리스타", "대포", "포탑 화살탑")):
        print(f"  {name:6s}: " + " | ".join(f"등급{t + 1} {dmg[k, t] / max(tn[k, t], 1):6.1f} ({tn[k, t]})" for t in range(4)))
    share = dmg.sum(1) / dmg.sum()
    print("  전체 피해 비중: " + ", ".join(f"{nm} {share[k]:.0%}" for k, nm in enumerate(("화살탑", "발리스타", "대포", "포탑"))))


if __name__ == "__main__":
    main()
