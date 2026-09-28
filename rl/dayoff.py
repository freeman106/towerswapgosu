"""휴식일 개입: ① 탐색 에이전트 기록에서 휴식일 제안(단계 108)에 어떻게 답했고 그날 무엇을 했는지, 사망일과 함께 본다.
② 제안 직전 상태로 되돌려 가상 난수 여러 개로 복제한 뒤, 휴식일 처리만 바꾸고 이후는 같은 탐색 에이전트로 이어 둔다.
  지금 방식(탐색이 결정) / 거절 / 수락 후 즉시 Done / 수락 후 그날은 정책만 / 수락 후 탐색
  --no-toss: 이어 두는 동안 휴식일 버리기를 행동 마스크로 막는다
사용: python rl/dayoff.py rl/runs/long1/agent_40M.pt [--pool 96] [--copies 8] [--days 31] [--no-toss]"""
import argparse
import json
import os
import pickle
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from search_eval import Policy, act_cat, board_info, play
from rescue import restore

OFFER, YES, NO, DONE = 108, 216, 217, 224
SEARCH = (3, 2, 8)  # 기존 탐색: 후보 상위 3 + 무작위 2, 난수 8묶음


def towers(cells):
    return sum(1 for c in cells if c[:1] in "tbcw")


def offer_report(frames, final):
    """기록에서 휴식일 제안·답·그날 행동을 요약하고 제안 지점(게임, 프레임 번호, 일차)을 돌려준다"""
    offers = []
    for i, fr in enumerate(frames):
        for t, f in enumerate(fr):
            if f["p"] != OFFER:
                continue
            day = f["d"]
            took = f["a"] == YES
            row = {"game": i, "t": t, "day": day, "took": took, "base": f.get("base", f["a"]), "final": int(final[i]),
                   "hearts": f["h"], "towers0": towers(f["b"])}
            if took:
                rest = [g for g in fr[t + 1:] if g["d"] == day]
                cells = lambda g: [g["b"][r * 6:(r + 1) * 6] for r in range(7)]
                cats = Counter(act_cat(cells(g), g["p"], g["a"]) for g in rest)
                row.update(actions=len(rest), toss=cats["버리기"], towers1=towers(rest[-1]["b"]) if rest else row["towers0"],
                           cats=dict(cats.most_common(6)))
            offers.append(row)
    print(f"\n[휴식일 제안] {len(offers)}건 · 게임 {len(set(o['game'] for o in offers))}판")
    for day in sorted(set(o["day"] for o in offers)):
        os_ = [o for o in offers if o["day"] == day]
        took = [o for o in os_ if o["took"]]
        flip = sum(o["base"] != (YES if o["took"] else NO) for o in os_)
        died = sum(o["final"] == day for o in took)
        print(f"  {day}일: 제안 {len(os_)} · 수락 {len(took)} (탐색이 정책의 답을 바꾼 것 {flip}) · 수락 후 그날 밤 사망 {died}"
              f" · 거절 후 그날 밤 사망 {sum(o['final'] == day for o in os_ if not o['took'])}")
        for o in took:
            print(f"    게임 {o['game']:2d}: 하트 {o['hearts']:2d} · 정책 답 {'수락' if o['base'] == YES else '거절'} · 행동 {o['actions']} · "
                  f"버리기 {o['toss']} · 방어물 {o['towers0']} → {o['towers1']} · 사망일 {o['final']} · {o['cats']}")
    return offers


class DayOffControl:
    """play()의 ctl: 휴식일 제안과 그날의 행동을 조건에 맞게 바꾸고, 제안일의 밤 직전 보드를 기록한다.
    mode: now(탐색·정책이 결정) / no(거절) / done(수락 후 즉시 Done) / policy(수락, 그날은 정책만) / search(수락, 그날도 탐색)"""

    def __init__(self, mode, n):
        self.mode = mode
        self.day0 = np.zeros(n, int)       # 제안일 (첫 호출 때 기록)
        self.took = np.zeros(n, bool)
        self.dusk = [None] * n             # 제안일 마지막 행동 직전 보드: (방어물 수, 점유율, 하트)
        self.toss = np.zeros(n, int)       # 제안일 버리기 수
        self.acts = np.zeros(n, int)       # 제안일 행동 수

    def before(self, env, alive):
        forced, nosearch = {}, np.zeros(len(alive), bool)
        for i in np.nonzero(alive)[0]:
            day, hearts, _, phase, _ = env.state(int(i))
            if self.day0[i] == 0:
                self.day0[i] = day
            if day != self.day0[i]:
                continue
            if phase == OFFER and self.mode != "now":
                forced[i] = NO if self.mode == "no" else YES
            elif self.took[i] and self.mode == "done":
                forced[i] = DONE
            elif self.took[i] and self.mode == "policy":
                nosearch[i] = True
        return forced, nosearch

    def after(self, env, i, a):
        day, hearts, _, phase, _ = env.state(int(i))
        if day != self.day0[i]:
            return
        cells, tur = env.board(int(i))
        if phase == OFFER:
            self.took[i] = a == YES
            return
        self.acts[i] += 1
        self.toss[i] += act_cat(cells, phase, a) == "버리기"
        occ, _ = board_info(cells, tur)
        self.dusk[i] = (towers([c for row in cells for c in row]), occ, hearts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--pool", type=int, default=96)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--copies", type=int, default=8)
    p.add_argument("--days", default="31", help="되돌릴 제안일 (쉼표로 여러 개)")
    p.add_argument("--modes", default="now,no,done,policy,search")
    p.add_argument("--no-toss", action="store_true", help="이어 두는 환경에 휴식일 버리기 금지 (탐색 굴리기에도 적용)")
    p.add_argument("--frames", default="", help="기록 캐시(pickle). 있으면 다시 두지 않는다")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n, M = args.pool, args.copies

    # 1. 탐색 에이전트 기록 (캐시)
    if args.frames and os.path.exists(args.frames):
        with open(args.frames, "rb") as f:
            frames, final = pickle.load(f)
    else:
        t0 = time.time()
        frames = [[] for _ in range(n)]
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        final = play(pol, n, args.seed, True, *SEARCH, st, 2.0, frames=frames)
        print(f"탐색 에이전트 {n}판: day {final.mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
        if args.frames:
            with open(args.frames, "wb") as f:
                pickle.dump((frames, final), f)
    print("사망일 분포:", dict(sorted(Counter(int(d) for d in final).items())))
    offers = offer_report(frames, final)

    # 2. 제안 직전 상태로 되돌려 조건별로 이어 두기
    days = {int(x) for x in args.days.split(",")}
    points = [(o["game"], o["t"]) for o in offers if o["day"] in days]
    if not points:
        return
    P = len(points)
    print(f"\n되돌릴 제안 {P}개 × 복제 {M}", flush=True)
    hold = restore(frames, args.seed, points, M)
    res = {}
    for mode in args.modes.split(","):
        t0 = time.time()
        e2 = ts.VecEnv(P * M, seed=1, no_toss_day_off=args.no_toss)
        e2.load_from(hold, np.arange(P * M, dtype=np.int64), np.arange(P * M, dtype=np.int64))
        ctl = DayOffControl(mode, P * M)
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        fin = play(pol, P * M, args.seed + 7, True, *SEARCH, st, 2.0, env=e2, ctl=ctl)
        res[mode] = {"final": fin.tolist(), "day0": ctl.day0.tolist(), "took": ctl.took.tolist(), "toss": ctl.toss.tolist(),
                     "acts": ctl.acts.tolist(), "dusk": ctl.dusk}
        D = ctl.day0
        dk = np.array([d if d is not None else (np.nan,) * 3 for d in ctl.dusk], float)
        print(f"  {mode:7s}: 수락 {ctl.took.mean():4.0%} · 그 밤 생존 {np.mean(fin > D):5.1%} · 평균 최종 {fin.mean():.2f}일 · "
              f"35일 도달 {np.mean(fin >= 35):4.0%} · 40일 {np.mean(fin >= 40):4.0%} · 밤 직전 방어물 {np.nanmean(dk[:, 0]):.1f} "
              f"(점유율 {np.nanmean(dk[:, 1]):.0%}, 하트 {np.nanmean(dk[:, 2]):.1f}) · 그날 행동 {ctl.acts.mean():.0f} · 버리기 {ctl.toss.mean():.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"points": points, "offers": offers, "res": res}, f)


if __name__ == "__main__":
    main()
