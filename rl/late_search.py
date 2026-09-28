"""후반 탐색 짝 비교: 휴식일 수정(버리기 금지, 휴식일에는 탐색 끔)을 적용한 탐색 에이전트로 게임을 두고, 31일 아침 상태를 모아
가상 난수 여러 개로 복제한 뒤 이후 평일에 탐색을 켠(ON) 조건과 끈(OFF) 조건으로 이어 둔다.
휴식일 제안과 휴식일 행동은 양쪽 모두 정책만(같은 잡음) + 버리기 금지로 똑같이 처리한다.
보고: 생존일, 35일 도달률, 하루 하트 회복량(밤 직전 − 아침), 밤 피해량(밤 직전 − 다음 날 아침, 사망한 밤은 밤 직전 하트 전부)
사용: python rl/late_search.py rl/runs/long1/agent_40M.pt [--pool 96] [--copies 8]"""
import argparse
import json
import os
import pickle
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from dayoff import OFFER, SEARCH
from rescue import restore
from search_eval import NoSearchDayOff, Policy, play


class LateCtl:
    """play()의 ctl: 휴식일 제안·휴식일에는 탐색하지 않고(mode="off"면 항상 탐색 안 함), 날마다 하트를 기록한다"""

    def __init__(self, mode, n):
        self.mode = mode
        self.day = np.zeros(n, int)
        self.morning = np.zeros(n, int)
        self.last = np.zeros(n, int)
        self.off = np.zeros(n, bool)
        self.rec = []  # (게임, 일차, 아침 하트, 밤 직전 하트, 다음 날 아침 하트(사망이면 0), 휴식일 여부)

    def before(self, env, alive):
        nosearch = np.zeros(len(alive), bool)
        for i in np.nonzero(alive)[0]:
            day, hearts, _, phase, _ = env.state(int(i))
            if day != self.day[i]:
                if self.day[i]:
                    self.rec.append((int(i), int(self.day[i]), int(self.morning[i]), int(self.last[i]), hearts, bool(self.off[i])))
                self.day[i], self.morning[i] = day, hearts
            self.off[i] = day % 10 == 1 and env.made(int(i))[2] == day
            self.last[i] = hearts
            nosearch[i] = self.mode == "off" or phase == OFFER or self.off[i]
        return {}, nosearch

    def after(self, env, i, a):
        pass

    def finish(self):
        """마지막 날(사망한 밤)을 기록에 넣는다"""
        for i in np.nonzero(self.day)[0]:
            self.rec.append((int(i), int(self.day[i]), int(self.morning[i]), int(self.last[i]), 0, bool(self.off[i])))


def heart_table(rec, days):
    """일차별 평균: 낮 하트 회복(휴식일 제외), 밤 피해, 표본 수"""
    out = {}
    for d in days:
        r = [x for x in rec if (x[1] == d if d else True)]
        wk = [x[3] - x[2] for x in r if not x[5]]
        out[d] = (np.mean(wk) if wk else np.nan, np.mean([x[3] - x[4] for x in r]) if r else np.nan, len(r))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--pool", type=int, default=96)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--copies", type=int, default=8)
    p.add_argument("--day", type=int, default=31)
    p.add_argument("--frames", default="runs/late_frames4242.pkl", help="풀 기록 캐시(pickle)")
    p.add_argument("--out", default="runs/late_search1.json")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n, M, D = args.pool, args.copies, args.day

    # 1. 휴식일 수정을 적용한 탐색 에이전트로 풀을 둔다 (캐시)
    if os.path.exists(args.frames):
        with open(args.frames, "rb") as f:
            frames, final = pickle.load(f)
    else:
        t0 = time.time()
        frames = [[] for _ in range(n)]
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        final = play(pol, n, args.seed, True, *SEARCH, st, 2.0, frames=frames, ctl=NoSearchDayOff(), no_toss=True)
        print(f"풀 {n}판 (탐색 + 휴식일 수정): day {final.mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
        with open(args.frames, "wb") as f:
            pickle.dump((frames, final), f)
    print("풀 사망일 분포:", dict(sorted(Counter(int(d) for d in final).items())))

    # 2. 31일 아침 상태 (그날 첫 행동 직전)
    points = []
    for i, fr in enumerate(frames):
        t = next((k for k, f in enumerate(fr) if f["d"] == D), None)
        if t is not None:
            points.append((i, t))
    P = len(points)
    offers = sum(frames[i][t]["p"] == OFFER for i, t in points)
    print(f"{D}일 아침 상태 {P}개 (그중 휴식일 제안 {offers}개) × 복제 {M}", flush=True)
    hold = restore(frames, args.seed, points, M)

    # 3. 평일 탐색 ON / OFF로 이어 둔다 (같은 상태·가상 난수·정책 잡음)
    fin, ctls, secs = {}, {}, {}
    for mode in ("on", "off"):
        t0 = time.time()
        e2 = ts.VecEnv(P * M, seed=1, no_toss_day_off=True)
        e2.load_from(hold, np.arange(P * M, dtype=np.int64), np.arange(P * M, dtype=np.int64))
        ctls[mode] = LateCtl(mode, P * M)
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        fin[mode] = play(pol, P * M, args.seed + 7, mode == "on", *SEARCH, st, 2.0, env=e2, ctl=ctls[mode]).reshape(P, M)
        ctls[mode].finish()
        secs[mode] = time.time() - t0
        print(f"  {mode}: {secs[mode]:.0f}s · 탐색한 결정 {st['searched']} · 바꾼 선택 {st['changed']}", flush=True)

    # 4. 보고
    on, off = fin["on"], fin["off"]
    d = (on - off).astype(float)
    per_state = d.mean(1)
    se = per_state.std(ddof=1) / np.sqrt(P)
    print(f"\n[{D}일 아침 {P}개 상태 × 복제 {M} = {P * M}판, 짝 비교]")
    print(f"  생존일(최종 일차): ON {on.mean():.2f} · OFF {off.mean():.2f} · 차이(ON − OFF) {d.mean():+.2f} ± {se:.2f}(상태 단위 표준오차)"
          f" · ON이 더 오래 {np.mean(d > 0):.0%} / 같음 {np.mean(d == 0):.0%} / 더 짧게 {np.mean(d < 0):.0%}"
          f" · 상태별 평균 차이 > 0인 상태 {np.mean(per_state > 0):.0%}")
    for x in (32, 33, 35, 40):
        print(f"  {x}일 도달: ON {np.mean(on >= x):5.1%} · OFF {np.mean(off >= x):5.1%}")
    print("  하트 (낮 회복 = 밤 직전 − 아침, 휴식일 제외 / 밤 피해 = 밤 직전 − 다음 아침, 사망한 밤은 남은 하트 전부):")
    tabs = {m: heart_table(ctls[m].rec, [0, D, D + 1, D + 2, D + 3]) for m in ("on", "off")}
    for dd in (D, D + 1, D + 2, D + 3, 0):
        a, b = tabs["on"][dd], tabs["off"][dd]
        name = "전체(날 평균)" if dd == 0 else f"{dd}일"
        print(f"    {name:10s}: 낮 회복 ON {a[0]:+.2f} · OFF {b[0]:+.2f} | 밤 피해 ON {a[1]:.2f} · OFF {b[1]:.2f} | 표본(날) ON {a[2]} · OFF {b[2]}")
    print(f"  계산 시간: ON {secs['on']:.0f}s · OFF {secs['off']:.0f}s")
    with open(args.out, "w") as f:
        json.dump({"points": points, "final": {m: fin[m].tolist() for m in fin}, "hearts": {m: ctls[m].rec for m in ctls}}, f)


if __name__ == "__main__":
    main()
