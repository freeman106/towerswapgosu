"""계획 선택기 데이터: teacher.py 실행의 결정 기록을 재생해(교사 굴림 없이, distill_data.Replay와 같은 방식) 교사가 평가한
모든 결정 상태에서 후보를 같은 상태·같은 기준 행동으로 다시 만든다(VecEnv.teacher_plans + teacher.py와 같은 중복 제거·정렬·최대 6개).
확인: 기록된 결정마다 보드, 기준 행동, 후보 수·순서·수 설명·결과 무기·사라진 무기·스왑이 기록과 같은지, 그리고 교사의 기록 규칙
(교체는 전부, 유지는 평가 순번 % 25 == 0일 때만)이 재생한 평가 순번과 맞는지. 최종 일차도 원래 실행과 같아야 한다.
결정 하나 = 관측(격자·스칼라), 기준 행동과 그 가상 결과(VecEnv.action_effects), 후보별 첫·둘째 수·정책 로그 확률·결과(teacher_plans).
정답(입력 아님): 후보별 교사 기준 통과(기록된 결정은 판정, 기록 안 된 결정은 교사가 유지했으므로 전부 불통과),
  후보별 ΔQ·표준오차(기록된 결정만, 나머지는 NaN). 유지 결정은 1/25만 기록됐다(ΔQ 가중치 25로 보정할 수 있게 logged·replaced 표시).
사용: python rl/planner_data.py --tag s12345 --games 1024 --seed 12345 --dec runs/teacher1024_s12345_decisions.json --games-json runs/teacher1024_s12345_games.json"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_bronze_flow import run
from eval_restrict import CONDS, Restrict
from force_silver_prep import SEEDS
from ppo import A, C, H, S, W, parse_emergency
from search_eval import Policy
from teacher import adesc, drag_xy, swap_key

NC = 6        # 후보 최대 수 (teacher.py ncand)
NCR = 6       # 결과 무기 최대 기록 수
KIND = {"t": 0, "b": 1, "c": 2, "w": 3}


def teacher_cands(L, base, lg, ncand=NC):
    """teacher.py Teacher.force와 같은 후보 정리"""
    seen, cands = set(), []
    for p in sorted(L, key=lambda p: (p[1] >= 0, -lg[p[0]])):
        key = (swap_key(p[0]), swap_key(p[1]), tuple(sorted(p[3])))
        if key in seen:
            continue
        seen.add(key)
        if p[1] < 0 and swap_key(p[0]) == swap_key(base):
            continue
        cands.append(p)
        if len(cands) >= ncand:
            break
    return cands


def enc_created(created):
    """결과 무기 (종류, 등급, x, y) 목록 → [NCR, 4] (없으면 -1)"""
    out = np.full((NCR, 4), -1, np.int8)
    for j, (k, t, x, y) in enumerate(created[:NCR]):
        out[j] = (KIND[k], t, x, y)
    return out


def enc_lost(lost):
    """사라진 무기 (종류, 등급) 목록 → [종류 4 × 등급 4] 수"""
    out = np.zeros(16, np.int8)
    for k, t in lost:
        out[KIND[k] * 4 + min(t, 4) - 1] += 1
    return out


class PlanReplay(Restrict):
    def __init__(self, n, dec):
        super().__init__(n)
        self.declog = {(d["game"], d["step"]): d for d in dec}  # self.log는 상자 합성 계획(Ext)이 쓴다
        self.pending = [None] * n
        self.counter = 0
        self.check = Counter()
        self.og = np.zeros((n, C, H, W), np.float32)
        self.os = np.zeros((n, S), np.float32)
        self.om = np.zeros((n, A), bool)
        self.R = {k: [] for k in ("grid", "scal", "game", "step", "day", "hearts", "swaps", "base", "lp_base", "b_ok", "b_same", "b_sig",
                                  "b_cr", "b_lost", "b_res", "b_dsw", "k", "a1", "a2", "lp1", "c_sig", "c_cr", "c_lost", "c_res",
                                  "passed", "dq", "dse", "logged", "replaced", "chosen")}

    def force(self, env, alive, logits, noise, act):
        forced = super().force(env, alive, logits, noise, act)  # 상자 합성 계획(원래 실행과 같음)
        live = [int(i) for i in np.nonzero(alive)[0]]
        recheck = []
        for i in live:
            pend = self.pending[i]
            if pend is None:
                continue
            if i in forced or self.cur[i]["day"] != pend["day"]:
                assert pend["done"] is None, f"게임 {i}: 원래 실행은 계획을 끝까지 확인했는데 재생에서는 중단"
                self.pending[i] = None
            elif self.cur[i]["phase"] == 0:
                recheck.append(i)
        for i in recheck:
            pend = self.pending[i]
            self.pending[i] = None
            assert pend["done"] is not None, f"게임 {i}: 원래 실행은 중단했는데 재생에서는 재확인 상태"
            if pend["done"] and act[i] != pend["a2"]:
                forced[i] = pend["a2"]
        cand = [i for i in live if i not in forced and i not in recheck and self.pending[i] is None
                and self.cur[i]["phase"] == 0 and self.cur[i]["swaps"] >= 1
                and not (self.cur[i]["day"] % 10 == 1 and env.made(i)[2] == self.cur[i]["day"])]
        if not cand:
            return forced
        P = env.teacher_plans(np.array(cand, np.int64), SEEDS)
        jobs = []
        for i, L in zip(cand, P):
            cs = teacher_cands(L, int(act[i]), logits[i])
            if cs:
                jobs.append((i, int(act[i]), cs))
            elif (i, int(self.step[i])) in self.declog:
                raise AssertionError(f"게임 {i} 행동 {self.step[i]}: 기록된 결정인데 재생에서는 후보가 없음")
        if not jobs:
            return forced
        env.observe(self.og, self.os, self.om)
        eff = env.action_effects(np.array([j[0] for j in jobs], np.int64), np.array([j[1] for j in jobs], np.int64), SEEDS)
        lsm = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        for (i, base, cs), ef in zip(jobs, eff):
            self.counter += 1  # teacher.py의 '평가한 결정' 순번과 같아야 한다(같은 게임 순서)
            d = self.declog.get((i, int(self.step[i])))
            nc = len(cs)
            passed = np.full(NC, -1, np.int8)
            dq = np.full(NC, np.nan, np.float32)
            dse = np.full(NC, np.nan, np.float32)
            chosen = -1
            if d is None:
                assert self.counter % 25 != 0, f"게임 {i}: 평가 순번 {self.counter}은 기록돼야 하는데 기록이 없음"
                passed[:nc] = 0  # 교사가 유지(어떤 후보도 기준 통과 못 함)
                self.check["기록 없는 평가(유지)"] += 1
            else:
                cells = self.cur[i]["cells"] or env.board(i)[0]
                assert [x for row in env.board(i)[0] for x in row] == d["board"], f"게임 {i} 행동 {self.step[i]}: 보드가 기록과 다름"
                assert d["base"] == base and d["candidates"][0]["수"] == [adesc(cells, base)], f"게임 {i}: 기준 행동이 기록과 다름"
                recs = d["candidates"][1:]
                assert len(recs) == nc, f"게임 {i} 행동 {self.step[i]}: 후보 수 {nc} (기록 {len(recs)})"
                for c, (p, r) in enumerate(zip(cs, recs)):
                    desc = [adesc(cells, p[0])] + ([f"(준비 뒤) {drag_xy(p[1])}"] if p[1] >= 0 else [])
                    assert r["수"] == desc and r["결과 무기"] == [list(x) for x in p[3]] and r["사라진 무기"] == [list(x) for x in p[4]] \
                        and r["스왑"] == (2 if p[1] >= 0 else 1), f"게임 {i} 행동 {self.step[i]} 후보 {c}: 기록과 다름 {r['수']} / {desc}"
                    passed[c] = r["판정"] == "통과"
                    dq[c], dse[c] = r["ΔQ"], r["±"]
                self.check["후보 일치 결정"] += 1
                if d["chosen"]:
                    a1, a2 = d["chosen"]
                    chosen = next(c for c, p in enumerate(cs) if p[0] == a1 and p[1] == a2)
                    assert passed[chosen] == 1 and dq[chosen] == np.nanmax(np.where(passed[:nc] == 1, dq[:nc], np.nan)), f"게임 {i}: 교체 후보 판정 불일치"
                    if a1 != base:
                        forced[i] = a1
                    if a2 >= 0:
                        self.pending[i] = {"a2": a2, "done": d.get("완성"), "day": self.cur[i]["day"]}
                    self.check["교체 결정"] += 1
                else:
                    assert self.counter % 25 == 0 and not (passed[:nc] == 1).any(), f"게임 {i}: 유지 기록인데 순번 {self.counter} 또는 통과 후보"
                    self.check["기록된 유지 결정"] += 1
            R = self.R
            R["grid"].append(self.og[i].astype(np.float16))
            R["scal"].append(self.os[i].astype(np.float16))
            for k, v in (("game", i), ("step", int(self.step[i])), ("day", self.cur[i]["day"]), ("hearts", self.cur[i]["hearts"]),
                         ("swaps", self.cur[i]["swaps"]), ("base", base), ("lp_base", float(lsm[i, base])), ("k", nc),
                         ("logged", d is not None), ("replaced", bool(d and d["chosen"])), ("chosen", chosen)):
                R[k].append(v)
            ok, same, sig, cr, lost, res, dsw = ef
            R["b_ok"].append(ok)
            R["b_same"].append(same)
            R["b_sig"].append(np.array(sig, np.int8))
            R["b_cr"].append(enc_created(cr))
            R["b_lost"].append(enc_lost(lost))
            R["b_res"].append(res)
            R["b_dsw"].append(dsw)
            a1 = np.full(NC, -1, np.int16)
            a2 = np.full(NC, -1, np.int16)
            lp1 = np.full(NC, np.nan, np.float32)
            csig = np.zeros((NC, 12), np.int8)
            ccr = np.full((NC, NCR, 4), -1, np.int8)
            clost = np.zeros((NC, 16), np.int8)
            cres = np.zeros(NC, np.float32)
            for c, p in enumerate(cs):
                a1[c], a2[c], lp1[c] = p[0], p[1], lsm[i, p[0]]
                csig[c], ccr[c], clost[c], cres[c] = p[2], enc_created(p[3]), enc_lost(p[4]), p[5]
            for k, v in (("a1", a1), ("a2", a2), ("lp1", lp1), ("c_sig", csig), ("c_cr", ccr), ("c_lost", clost), ("c_res", cres),
                         ("passed", passed), ("dq", dq), ("dse", dse)):
                R[k].append(v)
        return forced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--tag", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dec", required=True)
    p.add_argument("--games-json", required=True)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    dec = json.load(open(args.dec))
    want = np.array(json.load(open(args.games_json))["T"]["final"])
    t0 = time.time()
    c = PlanReplay(args.games, dec)
    fin, _, _ = run(pol, args.games, args.seed, rule, None, ctl=c)
    same = fin == want
    n_rep = sum(1 for d in dec if d["chosen"])
    print(f"재생 {time.time() - t0:.0f}s · 최종 일차 일치 {same.sum()}/{len(same)} · 평가 결정 {c.counter} · 확인 {dict(c.check)} · "
          f"기록 {len(dec)}개(교체 {n_rep}·유지 {len(dec) - n_rep})")
    assert same.all(), f"최종 일차가 다른 게임 {np.nonzero(~same)[0][:10]}"
    assert c.check["후보 일치 결정"] == len(dec), "기록된 결정을 모두 다시 찾지 못함"
    arr = {k: np.array(v) for k, v in c.R.items()}
    np.savez_compressed(f"runs/planner_{args.tag}.npz", seed=args.seed, **arr)
    pas = arr["passed"]
    print(f"저장 runs/planner_{args.tag}.npz · 결정 {len(arr['k'])} · 후보 {int((pas >= 0).sum())} (통과 {int((pas == 1).sum())}) · "
          f"ΔQ 아는 후보 {int((~np.isnan(arr['dq'])).sum())}")


if __name__ == "__main__":
    main()
