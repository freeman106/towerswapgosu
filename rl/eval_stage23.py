"""은상자 커리큘럼 2·3단계 평가 (정책 혼자, 학습에 쓰지 않은 상태 풀). 동상자를 놓은 상태에서 끝까지 둔다.
  bronze3: 동상자 3개, 교환 한 번이면 은상자 / bronze2: 동상자 2개(세 번째 동상자부터 만들어야 함)
기록: 은상자 합성·개봉 여부와 그때까지 쓴 스왑(휴식일 제외 드래그)·일수, 세 번째 동상자 생성(bronze2),
실패 이유(합성 전에 동상자를 잃음: 비상 개봉·버리기·판매·그 밖 / 동상자를 가진 채 사망 / 합성했지만 개봉 전 사망), 이후 생존.
사용: python rl/eval_stage23.py ckpt --layout bronze3 [--merge-rule 20,6]"""
import argparse
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, parse_emergency
from search_eval import DIRS, Policy, play
from starts import capture, describe

LEVEL = {"bronze3": 7, "bronze2": 8}


class BronzeStage:
    def __init__(self, n):
        self.start = np.zeros(n, int)
        self.spent = np.zeros(n, int)
        self.prev = [None] * n                 # 직전 상태의 (동상자 수, 은상자 수)
        self.last_cat = [""] * n              # 직전 행동이 동상자에 한 일
        self.made = [None] * n                 # 은상자 합성 (스왑, 날)
        self.opened = [None] * n               # 은상자 개봉 (스왑, 날)
        self.lost = [None] * n                 # 합성 전 첫 동상자 손실 이유

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        i = int(i)
        day, _, _, phase, _ = env.state(i)
        if self.start[i] == 0:
            self.start[i] = day
        cells = env.board(i)[0]
        flat = [c for row in cells for c in row]
        cnt = (flat.count("h2"), flat.count("h3") + flat.count("h4"))
        if self.prev[i] is not None:
            b0, s0 = self.prev[i]
            if cnt[1] > s0 and self.made[i] is None:
                self.made[i] = (int(self.spent[i]), day - self.start[i])
            elif cnt[0] < b0 and cnt[1] <= s0 and self.made[i] is None and self.lost[i] is None:
                self.lost[i] = self.last_cat[i] or "그 밖"
        self.prev[i] = cnt
        self.last_cat[i] = ""
        if a < 168 and phase == 0:
            if not (day % 10 == 1 and env.made(i)[2] == day):
                self.spent[i] += 1
            c = a // 4
            x, y = c % 6 + 1, c // 6 + 1
            dx, dy = DIRS[a % 4]
            tx, ty = x + dx, y + dy
            if cells[y - 1][x - 1] == "h2" and not (1 <= tx <= 6 and 1 <= ty <= 7 and cells[ty - 1][tx - 1] != "~~"):
                self.last_cat[i] = "동상자 버림"
        elif 168 <= a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            cell = cells[y - 1][x - 1] if y >= 1 else ""
            if phase in (0, 25) and cell == "h2":
                self.last_cat[i] = "동상자 개봉(비상)"
            elif phase in (0, 25) and cell in ("h3", "h4") and self.opened[i] is None and self.made[i] is not None:
                self.opened[i] = (int(self.spent[i]), day - self.start[i])
            elif phase == 83 and cell == "h2":
                self.last_cat[i] = "동상자 판매(상인)"
            elif phase == 116 and cell == "h2":
                self.last_cat[i] = "TNT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--layout", choices=list(LEVEL), required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--open-min-tier", type=int, default=3)
    p.add_argument("--emergency", default="2,0,5")
    p.add_argument("--merge-rule", default="20,6")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    hold, k, info = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
    rule = {"open_min_tier": args.open_min_tier, "emergency": parse_emergency(args.emergency),
            "merge_rule": tuple(int(x) for x in args.merge_rule.split(",")) if args.merge_rule else None}
    n = args.games
    env = ts.VecEnv(n, seed=args.seed, no_toss_day_off=True, **rule)
    added = env.add_starts_layout(hold, np.arange(k, dtype=np.int64), LEVEL[args.layout], args.layout, seed=args.pool_seed)
    env.set_start_frac(1.0)
    env.reset(np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool))
    c, fins = BronzeStage(n), []
    fin = play(Policy(args.ckpt, dev), n, args.seed, False, 0, 0, 0, None, env=env, ctl=c, fin_rec=fins)
    fins.sort(key=lambda f: f["env"])
    life = fin - c.start
    made = np.array([m is not None for m in c.made])
    opened = np.array([o is not None for o in c.opened])
    print(f"=== {args.ckpt} · {args.layout} ({added}상태, {describe(info)}) · 합성 우선 {args.merge_rule or '없음'} · {n}판 ===")
    print(f"  이후 생존 {life.mean():.2f}일 · 은상자 합성 {made.mean():.1%} · 합성한 은상자 개봉 {opened.mean():.1%}")
    if made.any():
        ms, md = np.array([m[0] for m in c.made if m]), np.array([m[1] for m in c.made if m])
        print(f"    합성까지: 스왑 평균 {ms.mean():.1f} (중앙 {np.median(ms):.0f}) · 일수 평균 {md.mean():.1f} (당일 {np.mean(md == 0):.0%})")
    if opened.any():
        os_, od = np.array([o[0] for o in c.opened if o]), np.array([o[1] for o in c.opened if o])
        print(f"    개봉까지: 스왑 평균 {os_.mean():.1f} (중앙 {np.median(os_):.0f}) · 일수 평균 {od.mean():.1f} (당일 {np.mean(od == 0):.0%})")
    if args.layout == "bronze2":
        third = np.array([f["made"][1] > 0 for f in fins])
        print(f"  세 번째 동상자(시작 뒤 합성으로 만든 동상자) 생성 {third.mean():.1%}")
    reinv = np.array([f["reinvest_opens"][2] for f in fins])
    print(f"  합성 은상자 개봉 {np.mean(reinv > 0):.1%} (판당 {reinv.mean():.2f}, 재투자 보상 대상)")
    cls = []
    for i in range(n):
        if opened[i]:
            cls.append("합성·개봉 성공")
        elif made[i]:
            cls.append("합성했지만 개봉 전 사망")
        elif c.lost[i]:
            cls.append(f"합성 전 동상자 잃음({c.lost[i]})")
        else:
            cls.append("동상자를 가진 채 합성 없이 사망")
    print("  결과 분류:")
    for k_, v in Counter(cls).most_common():
        print(f"    {k_}: {v / n:.1%} (이후 생존 {np.mean([life[i] for i in range(n) if cls[i] == k_]):.1f}일)")

if __name__ == "__main__":
    main()
