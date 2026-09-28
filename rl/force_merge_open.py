"""은상자 개봉일 강제 합성 짝 비교 (1단계 평가 풀, 같은 상태·같은 난수, 정책 혼자).
조건: 정책 그대로 / 기본→동 합성 개입 / 동→은 합성 개입.
개입: 은상자를 연 당일, 남은 스왑이 --min-swaps 이상이고 그날 개입이 --max-k번 미만일 때, 그 합성이 가능한 드래그
(복제본에 적용해 동 무기 또는 은 이상 무기가 실제로 생기는 수, 엔진 판정)가 있고 정책이 고른 수가 그 합성이 아니면
후보끼리 정책 확률을 다시 정규화해 고른다(같은 샘플링 잡음으로 argmax). 정책이 스스로 합성하면 개입으로 세지 않는다.
보고: 개봉일 고른 수(생산·합성·성 보수·버리기), 밤 직전 공격 타워 종류·등급·5~6행, 그날 밤 피해, 동·은상자 생성,
두 번째 은상자 개봉, 개봉 뒤 생존. 전체 평균과, 개입이 일어난 게임만의 짝 비교(정책 그대로 대비 차이 ± 표준오차).
사용: python rl/force_merge_open.py ckpt [ckpt ...] [--games 512] [--min-swaps 20] [--max-k 3]"""
import argparse
from collections import Counter

import numpy as np
import torch

from diag_open_day import CATS, category
from eval_stage1 import run
from ppo import parse_emergency
from search_eval import Policy
from starts import capture

MODES = ((None, "정책 그대로"), ("bronze", "기본→동 개입"), ("silver", "동→은 개입"))
KINDS = ("t", "b", "c")  # 화살탑·발리스타·대포


def dusk_layout(cells, turrets):
    """공격 타워 종류 × 등급(1..4) 수, 5~6행 등급별 수, 기본 환산 전력"""
    kt = np.zeros((3, 4), int)
    bottom = np.zeros(4, int)
    for r, row in enumerate(cells, start=1):
        for c in row:
            if c[:1] in KINDS:
                t = min(int(c[1]), 4) - 1
                kt[KINDS.index(c[0]), t] += 1
                bottom[t] += r in (5, 6)
    for c in turrets:
        if c[:1] == "t":
            kt[0, min(int(c[1]), 4) - 1] += 1
    return {"kt": kt, "bottom": bottom, "strength": int(sum(kt[:, t].sum() * 3 ** t for t in range(4)))}


class ForceMerge:
    def __init__(self, n, mode, min_swaps, max_k):
        self.mode, self.min_swaps, self.max_k = mode, min_swaps, max_k
        self.open_day = np.full(n, -1)
        self.k = np.zeros(n, int)
        self.cats = [Counter() for _ in range(n)]
        self.dusk = {}
        self.dusk_h, self.next_h = {}, {}
        self.silver_opens = [0] * n
        self.out = np.zeros((168, 8), np.int32)

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def choose(self, env, alive, logits, noise, act):
        forced = {}
        if self.mode is None:
            return forced
        for i in np.nonzero(alive)[0]:
            if self.open_day[i] < 0 or self.k[i] >= self.max_k:
                continue
            day, _, swaps, phase, _ = env.state(int(i))
            if day != self.open_day[i] or phase != 0 or swaps < self.min_swaps:
                continue
            env.drag_outcomes(int(i), self.out)
            v = self.out
            if self.mode == "bronze":
                cand = (v[:, 1] > 0) & (v[:, 2] <= 0) & (v[:, 3] <= 0)
            else:
                cand = (v[:, 2] > 0) | (v[:, 3] > 0)
            idx = np.nonzero(cand)[0]
            if len(idx) == 0 or act[i] in idx:
                continue
            forced[int(i)] = int(idx[np.argmax(logits[i, idx] + noise[i, idx])])
            self.k[i] += 1
        return forced

    def after(self, env, i, a):
        i = int(i)
        day, hearts, _, phase, _ = env.state(i)
        od = self.open_day[i]
        if od >= 0 and day == od + 1 and i not in self.next_h:
            self.next_h[i] = hearts
        if phase not in (0, 25):
            return
        cells, tur = env.board(i)
        if 168 <= a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if y >= 1 and cells[y - 1][x - 1][:1] == "h" and int(cells[y - 1][x - 1][1]) >= 3:
                self.silver_opens[i] += 1
                if od < 0:
                    self.open_day[i] = od = day
        if day != od:
            return
        if a < 168 and phase == 0:
            env.drag_outcomes(i, self.out)
            self.cats[i][category(self.out[a])] += 1
        elif a == 224:
            self.dusk[i] = dusk_layout(cells, tur)
            self.dusk_h[i] = hearts


def metrics(fin, fins, c, i):
    od = c.open_day[i]
    d = c.dusk.get(i)
    m = {"생존": fin[i] - od, "밤 피해": c.dusk_h[i] - c.next_h.get(i, 0) if i in c.dusk_h else np.nan,
         "동상자": fins[i]["made"][1], "은상자": fins[i]["made"][2], "둘째 은상자": float(c.silver_opens[i] >= 2)}
    for k in CATS:
        m[k] = c.cats[i][k]
    if d:
        m["전력"] = d["strength"]
        for t, name in enumerate(("기본", "동", "은")):
            m[f"{name} 무기"] = d["kt"][:, t].sum()
        m["5~6행 동+"] = d["bottom"][1:].sum()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--min-swaps", type=int, default=20)
    p.add_argument("--max-k", type=int, default=3)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    hold, k, _ = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5")}
    n = args.games
    keys = ("생존", "밤 피해", "전력", "기본 무기", "동 무기", "은 무기", "5~6행 동+", "생산", "동 합성", "은 합성", "상자 합성", "성 보수",
            "버리기", "그 밖의 이동", "동상자", "은상자", "둘째 은상자")
    for ck in args.ckpts:
        pol = Policy(ck, dev)
        res = {}
        for mode, name in MODES:
            c = ForceMerge(n, mode, args.min_swaps, args.max_k)
            fin, fins, _ = run(pol, n, args.seed, hold, k, 3, rule, args.pool_seed, ctl=c)
            res[mode] = (c, [metrics(fin, fins, c, i) for i in range(n)])
        base = res[None][1]
        print(f"\n=== {ck} · 개봉일, 남은 스왑 ≥ {args.min_swaps}, 하루 최대 {args.max_k}회 · {n}판 ===")
        print("  전체 평균 (개봉일 고른 수는 판당 횟수, 무기·전력은 개봉일 밤 직전, 동·은상자는 게임 전체 생성):")
        print("    " + f"{'':<14s}" + "".join(f"{kk:>10s}" for kk in keys))
        for mode, name in MODES:
            ms = res[mode][1]
            row = [np.nanmean([m.get(kk, np.nan) for m in ms]) for kk in keys]
            print("    " + f"{name:<14s}" + "".join(f"{x:>10.2f}" for x in row))
        for mode, name in MODES[1:]:
            c, ms = res[mode]
            sel = np.nonzero(c.k > 0)[0]
            print(f"  [{name}] 개입이 일어난 게임 {len(sel)}/{n} (개입 {c.k.sum()}번, 게임당 {c.k[sel].mean() if len(sel) else 0:.1f}번) — 같은 게임의 정책 그대로 대비:")
            if len(sel) < 5:
                continue
            for kk in keys:
                a = np.array([ms[i].get(kk, np.nan) for i in sel], float)
                b = np.array([base[i].get(kk, np.nan) for i in sel], float)
                ok = ~np.isnan(a) & ~np.isnan(b)
                dd = a[ok] - b[ok]
                if len(dd) > 1:
                    print(f"    {kk:<10s} 개입 {a[ok].mean():7.2f} · 그대로 {b[ok].mean():7.2f} · 차이 {dd.mean():+6.2f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.2f} ({len(dd)}판)")


if __name__ == "__main__":
    main()
