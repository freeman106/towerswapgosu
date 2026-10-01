"""계획 선택기 실전 평가 (학습 없음). 원본 cn7_nf 20M + 기존 상자 개봉 규칙·무기 합성 우선·상자 합성 계획·대포 방향 금지는 그대로.
S: teacher.py와 같은 후보 생성(VecEnv.teacher_plans, 가상 판정 시드 — 실제 게임 난수와 분리)·같은 후보 정리·두 수 계획 재확인을 쓰고,
   밤까지 굴리는 교사 비교만 학습된 선택기로 바꾼다: 통과 확률 ≥ τ인 후보가 있으면 그중 예측 ΔQ 최대 계획, 없으면 원본 선택.
P: 원본 정책 그대로(같은 규칙·상자 계획). 같은 게임 시드·같은 샘플링 잡음으로 짝 비교.
시간: 후보 생성(teacher_plans, 재확인 포함) / 기준 행동 가상 결과(action_effects) / 특징 만들기 / 선택기 추론을 따로 잰다.
사용: python rl/planner_eval.py --games 256 --seed 2468 [--sel runs/planner_sel/selector.pt]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_bronze_flow import run
from eval_restrict import CONDS, report
from force_silver_prep import SEEDS
from planner import NC, features, load_selector
from planner_data import enc_created, enc_lost
from planner_train import select
from ppo import A, C, H, S, W, parse_emergency
from search_eval import Policy
from teacher import Teacher, extra_report, plan_kind


class Timed:
    """VecEnv 대리: teacher_plans 시간만 따로 잰다(나머지는 그대로 넘긴다)"""

    def __init__(self, env, t):
        self._env, self._t = env, t

    def teacher_plans(self, *a, **k):
        t0 = time.perf_counter()
        r = self._env.teacher_plans(*a, **k)
        self._t["후보 생성(teacher_plans)"] += time.perf_counter() - t0
        return r

    def __getattr__(self, name):
        return getattr(self._env, name)


class Sel(Teacher):
    def __init__(self, n, pol, seed, model, tau, dev):
        super().__init__(n, pol, seed, True)
        self.model, self.tau, self.dev = model, tau, dev
        self.og = np.zeros((n, C, H, W), np.float32)
        self.os = np.zeros((n, S), np.float32)
        self.om = np.zeros((n, A), bool)
        self.t = Counter()
        self.pmax = []

    def force(self, env, alive, logits, noise, act):
        return super().force(Timed(env, self.t), alive, logits, noise, act)

    def decide(self, env, jobs, step, forced, logits):
        t0 = time.perf_counter()
        env.observe(self.og, self.os, self.om)
        idx = np.array([j[0] for j in jobs], np.int64)
        base = np.array([j[1] for j in jobs], np.int64)
        eff = env.action_effects(idx, base, SEEDS)
        t1 = time.perf_counter()
        lsm = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        B = len(jobs)
        d = {"base": base, "lp_base": lsm[idx, base], "k": np.array([len(j[2]) for j in jobs]),
             "hearts": np.array([self.cur[i]["hearts"] for i in idx]), "day": np.array([self.cur[i]["day"] for i in idx]),
             "swaps": np.array([self.cur[i]["swaps"] for i in idx]),
             "b_ok": np.array([e[0] for e in eff]), "b_same": np.array([e[1] for e in eff]), "b_sig": np.array([e[2] for e in eff], np.int8),
             "b_cr": np.stack([enc_created(e[3]) for e in eff]), "b_lost": np.stack([enc_lost(e[4]) for e in eff]),
             "b_res": np.array([e[5] for e in eff], np.float32), "b_dsw": np.array([e[6] for e in eff], np.float32),
             "a1": np.full((B, NC), -1), "a2": np.full((B, NC), -1), "lp1": np.full((B, NC), np.nan, np.float32),
             "c_sig": np.zeros((B, NC, 12), np.int8), "c_cr": np.full((B, NC, 6, 4), -1, np.int8), "c_lost": np.zeros((B, NC, 16), np.int8),
             "c_res": np.zeros((B, NC), np.float32)}
        for b, (i, _, cs) in enumerate(jobs):
            for c, p in enumerate(cs):
                d["a1"][b, c], d["a2"][b, c], d["lp1"][b, c] = p[0], p[1], lsm[i, p[0]]
                d["c_sig"][b, c], d["c_cr"][b, c], d["c_lost"][b, c], d["c_res"][b, c] = p[2], enc_created(p[3]), enc_lost(p[4]), p[5]
        hand, cells, valid = features(d)
        t2 = time.perf_counter()
        with torch.no_grad():
            g = torch.from_numpy(self.og[idx]).to(self.dev)
            s = torch.from_numpy(self.os[idx]).to(self.dev)
            lp, dq = self.model(g, s, torch.from_numpy(hand).to(self.dev), torch.from_numpy(cells).to(self.dev))
            p, q = torch.sigmoid(lp).cpu().numpy(), dq.cpu().numpy()
        t3 = time.perf_counter()
        self.t["기준 행동 가상 결과(action_effects)"] += t1 - t0
        self.t["특징 만들기"] += t2 - t1
        self.t["선택기 추론"] += t3 - t2
        ch = select(p, q, valid, self.tau)
        for b, (i, base_a, cs) in enumerate(jobs):
            self.stats["평가한 결정"] += 1
            self.pmax.append(float(np.where(valid[b], p[b], 0).max()))
            day = self.cur[i]["day"]
            if ch[b] < 0:
                self.stats["유지(기존 선택)"] += 1
                continue
            chosen = cs[ch[b]]
            self.kinds[plan_kind(chosen)] += 1
            self.stats["교체"] += 1
            self.tday[i].add(day)
            if chosen[0] != base_a:
                forced[i] = int(chosen[0])
            else:
                self.stats["교체: 정책 첫 수 + 선택기 완성 수"] += 1
            if chosen[1] >= 0:
                self.pending[i] = (int(chosen[1]), list(chosen[2]), None, day)
                self.stats["두 수 계획 시작"] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--sel", default="runs/planner_sel/selector.pt")
    p.add_argument("--tau", type=float, default=None, help="기본: 학습에서 문턱 세트로 정한 값")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=2468)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    pol = Policy(args.ckpt, dev)
    model, ck = load_selector(args.sel, dev)
    tau = args.tau if args.tau is not None else ck["tau"]
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    res, secs = {}, {}
    for name in ("P", "S"):
        t0 = time.time()
        c = Teacher(args.games, pol, args.seed, False) if name == "P" else Sel(args.games, pol, args.seed, model, tau, dev)
        fin, fins, _ = run(pol, args.games, args.seed, rule, None, ctl=c)
        c.finish()
        res[name] = (fin, fins, c)
        secs[name] = time.time() - t0
        print(f"  {name}: {secs[name]:.0f}s", flush=True)
    labels = {"P": "P 원본 정책", "S": "S 계획 선택기"}
    print(f"\n=== {args.ckpt} · 정상 시작 {args.games}판 (시드 {args.seed}) · 대포 방향 금지 · 상자 합성 계획 켬 · 선택기 τ={tau} ===")
    report(res, args.games, labels, [("S", "P")])
    fin = {m: r[0] for m, r in res.items()}
    print(f"  30일 보스 통과: P {np.mean(fin['P'] > 30):.1%} · S {np.mean(fin['S'] > 30):.1%}")
    extra_report(res, args.games, [("S", "P")])
    c = res["S"][2]
    pm = np.array(c.pmax)
    tot = sum(c.t.values())
    print(f"\n  선택기 결정 {c.stats['평가한 결정']} · 개입 {c.stats['교체']} ({c.stats['교체'] / max(c.stats['평가한 결정'], 1):.1%}) · 종류 {dict(c.kinds)} · "
          f"최고 통과 확률 중앙 {np.median(pm):.3f}·90% {np.percentile(pm, 90):.3f} · 두 수 재확인 {dict((k, v) for k, v in c.stats.items() if k.startswith('두 수 계획:'))}")
    print(f"  실행 시간: P {secs['P']:.0f}s · S {secs['S']:.0f}s (S − P {secs['S'] - secs['P']:.0f}s) · S 안에서 "
          + " · ".join(f"{k} {v:.1f}s" for k, v in c.t.items()) + f" (합 {tot:.1f}s)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({m: {"final": r[0].tolist(), "tday": [sorted(x) for x in r[2].tday]} for m, r in res.items()} | {"secs": secs, "time": dict(c.t), "tau": tau}, f)


if __name__ == "__main__":
    main()
