"""동상자 → 은상자 흐름 진단 (추가 학습 없음, 한 수 은상자 합성 우선 켬 = force_chest_merge.Priority).
3단계(동상자 2개 배치, eval_stage23과 같은 평가 풀) 512판과 정상 시작 1,024판에서 게임마다 합성 은상자를 처음 만들 때까지 기록한다.
- 동시 보유: 행동 직전 상태의 보드에 동상자 2개·3개 이상이 함께 있던 적
- 연쇄 성공: 동상자가 새로 생기는 행동에서 바로 은상자가 됨(3개가 보드에 함께 보이지 않을 수 있다)
- 한 수 합성(3개 보유 중, 엔진 VecEnv.silver_merge_counts): 원래 게임 규칙상 드래그 한 번으로 은상자가 생김 /
  그중 지금 마스크에서 유효 / 그중 직후 개봉 가능(= 우선 규칙 후보). 밤 직전(스왑 0)은 패턴 검사(교환 한 번)로 본다
- 손실: 은상자 전에 보드의 동상자가 줄어듦(비상 개봉·버리기·상인 판매·TNT·그 밖), 그때 보유 수·일차·하트·스왑·보드
- 기존 지표 '세 번째 동상자 생성'(made[1] > 0)을 기존 두 개를 지킨 채 셋째를 만든 경우와 하나를 잃은 뒤 새로 만든 경우로 나눈다
사용: python rl/diag_bronze_flow.py rl/runs/cn7/agent_10M.pt [--out 접두사] [--replay 리플레이.json]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from eval_stage23 import LEVEL
from force_chest_merge import Priority
from ppo import A, C, H, S, W, parse_emergency
from record_dayoff_games import mark_off
from search_eval import DIRS, Policy, play
from starts import capture

BRONZE = 1 << 2  # 패턴 검사 비트: 동상자 합성


def bronze_cat(cells, phase, a):
    """행동 a가 동상자에 한 일 (없으면 빈 문자열)"""
    if a < 168:
        c = a // 4
        x, y = c % 6 + 1, c // 6 + 1
        dx, dy = DIRS[a % 4]
        tx, ty = x + dx, y + dy
        if phase == 0 and cells[y - 1][x - 1] == "h2" and not (1 <= tx <= 6 and 1 <= ty <= 7 and cells[ty - 1][tx - 1] != "~~"):
            return "버리기"
        return ""
    if a < 216:
        y, x = (a - 168) // 6, (a - 168) % 6 + 1
        if y < 1 or cells[y - 1][x - 1] != "h2":
            return ""
        return {0: "비상 개봉", 25: "비상 개봉", 83: "상인 판매", 116: "TNT", 43: "아이템 배치로 덮어씀"}.get(phase, "")
    return "TNT" if a == 223 else ""


def act_kind(phase, a):
    return "드래그" if a < 168 else "밤 시작(아침 해소 포함)" if a == 224 else "탭·선택" if a < 216 else "그 밖"


class Flow(Priority):
    """play()의 ctl: 한 수 은상자 합성 우선(Priority.choose)을 두면서 동상자 흐름을 기록한다"""

    def __init__(self, n):
        self.init_priority(n)
        self.step = np.zeros(n, int)
        self.start = np.zeros(n, int)
        self.cur = [None] * n   # 이번 상태
        self.prev = [None] * n  # 직전 행동 직전 상태
        self.last = [None] * n  # 직전 행동 (행동, 동상자에 한 일, 종류)
        self.pend = [None] * n  # 동상자 개봉 탭 뒤 아이템 창이 끝날 때까지 상자가 남아 있는 경우: 탭 시점 기록
        self.g = [{"held2": None, "held3": None, "max_b": 0, "losses": [], "new_bronze": [], "silver": None, "item_silver": 0,
                   "opened": None, "raw3": None, "mask3": None, "exec3": None, "blocked": Counter(), "hold3_states": 0,
                   "dist3": Counter(), "chain2_states": [], "forced": [], "end_b": 0, "streak": None,
                   "d2_ok": 0, "dusk_pend": None, "dusk_next": Counter()} for _ in range(n)]

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def choose(self, env, alive, logits, noise, act):
        self.env = env
        check = []
        for i in map(int, np.nonzero(alive)[0]):
            day, hearts, swaps, phase, _ = env.state(i)
            cnt, made, d1, _ = env.chest_info(i)
            b, s = cnt[2], cnt[3] + cnt[4]
            g, p = self.g[i], self.prev[i]
            if self.step[i] == 0:
                self.start[i] = day
            cells = env.board(i)[0] if b or s or (p and p["b"]) else None
            c = {"b": b, "s": s, "made": made, "day": day, "hearts": hearts, "swaps": swaps, "phase": phase, "cells": cells, "d1": d1}
            self.cur[i] = c
            if g["silver"] is not None:
                continue
            if p is not None:
                self.transition(i, p, c)
            if g["silver"] is not None:
                continue
            g["max_b"] = max(g["max_b"], b)
            g["end_b"] = b
            if b >= 2 and g["held2"] is None:
                g["held2"] = self.snap(i, c)
            if b >= 2 and phase == 0:
                check.append(i)
            elif b >= 3:
                self.hold3(i, c, None)
        if check:
            out = np.zeros((len(check), 3), np.int32)
            env.silver_merge_counts(np.array(check, np.int64), out)
            for i, r in zip(check, out):
                c = self.cur[i]
                if c["b"] >= 3:
                    self.hold3(i, c, r)
                elif r[0] > 0:
                    self.g[i]["chain2_states"].append(int(self.step[i]))
        forced = self.force(env, alive, logits, noise, act)
        for i, a in forced.items():
            self.g[i]["forced"].append((int(self.step[i]), int(act[i]), int(a)))
        return forced

    def force(self, env, alive, logits, noise, act):
        """강제 행동 (기본: 한 수 은상자 합성 우선). 하위 클래스가 바꾼다"""
        return Priority.choose(self, env, alive, logits, noise, act)

    def snap(self, i, c, r=None):
        f = {"step": int(self.step[i]), "day": c["day"], "rel_day": c["day"] - int(self.start[i]), "hearts": c["hearts"],
             "swaps": c["swaps"], "phase": c["phase"], "b": c["b"]}
        if r is not None:
            f["raw"], f["mask"], f["open"] = (int(x) for x in r)
        return f

    def hold3(self, i, c, r):
        """동상자 3개 이상 보유 상태: 한 수 합성 판정(입력 대기면 엔진 r, 밤 직전이면 패턴), 교환 거리(패턴)"""
        g = self.g[i]
        g["hold3_states"] += 1
        _, _, d1, d2 = self.env.chest_info(i, True)
        dist = 1 if d1 & BRONZE else 2 if d2 & BRONZE else 3
        g["dist3"][dist] += 1
        st = int(self.step[i])
        if g["held3"] is None:
            g["held3"] = {**self.snap(i, c, r), "dist": dist, "board": [x for row in c["cells"] for x in row]}
        p = self.prev[i]
        if p is None or p["b"] < 3:  # 3개 보유 구간 시작
            g["streak"] = {**self.snap(i, c, r), "dist": dist, "first_raw": None}
        if r is not None and r[0] > 0 and g["streak"]["first_raw"] is None:
            g["streak"]["first_raw"] = st
            g["streak"]["first_raw_via"] = self.last[i][2] if self.last[i] else ""
            g["streak"]["first_raw_day"] = c["day"]
        if r is not None and r[0] == 0 and dist == 2 and c["swaps"] >= 2:
            g["d2_ok"] += 1  # 교환 두 번 거리(패턴)이고 스왑 2개 이상 남은 입력 대기 상태
        if r is not None and g["dusk_pend"] is not None:  # 밤 직전 한 수 배치 → 다음 입력 대기
            g["dusk_next"]["다음 입력 대기에 한 수 유지" if r[0] > 0 else "다음 입력 대기에 배치 사라짐"] += 1
            g["dusk_pend"] = None
        if r is not None and r[0] > 0:
            g["raw3"] = g["raw3"] if g["raw3"] is not None else st
            if r[1] > 0:
                g["mask3"] = g["mask3"] if g["mask3"] is not None else st
            if r[2] > 0:
                g["exec3"] = g["exec3"] if g["exec3"] is not None else st
            elif r[1] == 0:
                g["blocked"]["마스크(규칙)에 없음"] += 1
            else:
                g["blocked"]["직후 개봉 불가"] += 1
        elif r is None and c["phase"] == 25 and d1 & BRONZE:
            g["blocked"]["밤 직전(스왑 0)에 한 수 배치"] += 1
            if g["dusk_pend"] is None:
                g["dusk_pend"] = st

    def transition(self, i, p, c):
        """직전 행동의 결과: 동상자 생성, 은상자 생성(연쇄·직접·아이템), 동상자 손실"""
        g = self.g[i]
        a, cat, kind = self.last[i]
        dm = [c["made"][t] - p["made"][t] for t in range(5)]
        st = int(self.step[i]) - 1
        if dm[2] > 0:
            g["new_bronze"].append({"step": st, "day": p["day"], "b_before": p["b"], "lost_before": len(g["losses"])})
        if dm[3] + dm[4] > 0 and c["s"] > p["s"]:
            typ = "연쇄" if dm[2] > 0 else "직접" if p["b"] - c["b"] >= 3 else "아이템"
            if typ == "아이템":
                g["item_silver"] += 1
                return
            if g["dusk_pend"] is not None:
                g["dusk_next"]["합성"] += 1
            g["silver"] = {**self.snap(i, p), "step": st, "type": typ, "via": kind, "a": a,
                           "chain2": st in g["chain2_states"], "held3_before": g["held3"] is not None, "streak": g["streak"]}
        elif c["b"] < p["b"]:
            # 개봉 탭 뒤 아이템 창을 거쳐 상자가 사라졌으면 탭 시점으로 기록한다
            pend = self.pend[i] if not cat else None
            src, st2, a2, cat2 = (pend["state"], pend["step"], pend["a"], pend["cat"]) if pend else (p, st, a, cat or "그 밖")
            g["losses"].append({**self.snap(i, src), "step": st2, "cause": cat2, "via": kind if not pend else "개봉 뒤 아이템 창", "a": a2,
                                "b": p["b"], "b_after": c["b"], "after_held3": g["held3"] is not None,
                                "board": [x for row in src["cells"] for x in row]})
            self.pend[i] = None
            if g["dusk_pend"] is not None:
                g["dusk_next"][f"잃음({cat2})"] += 1
                g["dusk_pend"] = None
        if c["phase"] in (0, 25) and c["b"] >= p["b"]:  # 손실 없이 입력 대기로 돌아옴
            self.pend[i] = None

    def after(self, env, i, a):
        i = int(i)
        c, g = self.cur[i], self.g[i]
        cat = bronze_cat(c["cells"], c["phase"], a) if c["cells"] else ""
        if g["silver"] is not None and g["opened"] is None and 168 <= a < 216 and c["phase"] in (0, 25):
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if y >= 1 and c["cells"][y - 1][x - 1] in ("h3", "h4"):
                g["opened"] = self.snap(i, c)
        self.last[i] = (a, cat, act_kind(c["phase"], a))
        if cat == "비상 개봉":
            self.pend[i] = {"state": c, "step": int(self.step[i]), "a": a, "cat": cat}
        self.prev[i] = c
        self.step[i] += 1


def run(pol, n, seed, rule, starts=None, frames=None, ctl=None):
    """starts = (hold, k, pool_seed)면 3단계(동상자 2개) 연습 시작, 아니면 정상 시작. ctl: Flow 하위 클래스 객체(없으면 Flow)"""
    c, fins = ctl or Flow(n), []
    if starts is not None:
        hold, k, pool_seed = starts
        env = ts.VecEnv(n, seed=seed, no_toss_day_off=True, **rule)
        env.add_starts_layout(hold, np.arange(k, dtype=np.int64), LEVEL["bronze2"], "bronze2", seed=pool_seed)
        env.set_start_frac(1.0)
        env.reset(np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool))
        fin = play(pol, n, seed, False, 0, 0, 0, None, env=env, ctl=c, fin_rec=fins, frames=frames)
    else:
        fin = play(pol, n, seed, False, 0, 0, 0, None, ctl=c, no_toss=True, fin_rec=fins, env_kw=rule, frames=frames)
    fins.sort(key=lambda f: f["env"])
    return fin, fins, c


def classify(g):
    """합성 은상자까지의 흐름에서 어디서 끊겼는지"""
    s = g["silver"]
    if s is not None:
        return f"은상자 합성({s['type']})" + (" → 개봉" if g["opened"] else " → 개봉 전 사망")
    if g["held3"] is not None:
        if g["exec3"] is not None:
            return "3개·합성 실행 가능했는데 안 됨"
        if g["raw3"] is not None:
            return "3개·한 수 배치 있었지만 마스크·개봉 조건으로 못 둠"
        if sum(g["blocked"].values()):
            return "3개·한 수 배치는 밤 직전(스왑 0)에만"
        return "3개 모았지만 한 수 배치 못 만듦"
    if g["losses"]:
        return "3개 전에 동상자 잃음"
    return "3개 전에 손실 없이 사망"


def pct(x, n):
    return f"{x / n:.1%} ({x})" if n else "-"


def dist(vals, fmt="{:.1f}"):
    v = np.array(vals, float)
    if not len(v):
        return "-"
    return f"평균 {fmt.format(v.mean())} · 중앙 {fmt.format(np.median(v))} · 10~90% {fmt.format(np.percentile(v, 10))}~{fmt.format(np.percentile(v, 90))}"


def report(title, fin, fins, c, n, stage3):
    G = c.g
    life = fin - c.start
    print(f"\n=== {title} ===")
    print(f"  이후 생존 {life.mean():.2f}일 · 우선 규칙 개입 판 {np.sum(c.k > 0)} (개입 {c.k.sum()}번, 정책이 스스로 한 수 합성 {np.sum(c.own > 0)}판)")

    # 1. '세 번째 동상자 생성' 집계 바로잡기
    old = np.array([f["made"][1] > 0 for f in fins])
    nb = [g["new_bronze"] for g in G]
    if stage3:
        print(f"\n  [1] 기존 지표 '세 번째 동상자 생성'(시작 뒤 동상자 생성 made[1] > 0): {pct(old.sum(), n)}")
        first = [e[0] if e else None for e in nb]
        keep = [i for i in range(n) if old[i] and first[i] and first[i]["b_before"] >= 2]
        relost = [i for i in range(n) if old[i] and first[i] and first[i]["b_before"] < 2]
        print(f"    · 기존 두 개를 지킨 채 새로 만듦(= 3개 동시 보유 또는 연쇄): {pct(len(keep), n)}")
        print(f"    · 기존 동상자를 잃은 뒤(1개 이하) 새로 만듦: {pct(len(relost), n)}"
              f" — 그중 뒤에 다시 3개 동시 보유 {sum(1 for i in relost if G[i]['held3'])}판·은상자 {sum(1 for i in relost if G[i]['silver'])}판")
        other = int(old.sum()) - len(keep) - len(relost)
        if other:
            print(f"    · 은상자 뒤에만 만듦: {pct(other, n)}")
        both = sum(1 for g in G if g["held3"] is not None or (g["silver"] and g["silver"]["type"] == "연쇄"))
        print(f"    → 바로잡은 지표 '동상자 3개 확보(동시 보유 또는 연쇄)': {pct(both, n)}")
    else:
        third = sum(1 for e in nb if any(x["b_before"] >= 2 for x in e))
        rebuild = sum(1 for e in nb if any(x["b_before"] < 2 and x["lost_before"] for x in e))
        print(f"\n  [1] 동상자 생성(은상자 전, 1개 이상) {pct(sum(1 for e in nb if e), n)} · 기존 지표 made[1] > 0 {pct(old.sum(), n)}")
        print(f"    · 2개를 가진 채 새로 만듦(세 번째 = 3개 동시 보유 또는 연쇄): {pct(third, n)} · 동상자를 잃은 뒤 다시 만듦: {pct(rebuild, n)}")

    # 2. 흐름 깔때기
    h2 = [g["held2"] is not None for g in G]
    h3 = [g["held3"] is not None for g in G]
    chain_only = [g["silver"] is not None and g["silver"]["type"] == "연쇄" and not g["silver"]["held3_before"] for g in G]
    raw3 = [g["raw3"] is not None for g in G]
    mask3 = [g["mask3"] is not None for g in G]
    exec3 = [g["exec3"] is not None for g in G]
    made = [g["silver"] is not None for g in G]
    opened = [g["opened"] is not None for g in G]
    typ = Counter(g["silver"]["type"] for g in G if g["silver"])
    print(f"\n  [2] 흐름 (게임 수, {n}판 기준)")
    print(f"    동상자 2개 동시 보유          {pct(sum(h2), n)}")
    at0 = sum(1 for g in G if g["held3"] and g["held3"]["step"] == 0)
    print(f"    3개 동시 보유                 {pct(sum(h3), n)}" + (f" — 그중 시작부터 3개 {at0}판(풀 상태에 동상자가 이미 있음)" if at0 else ""))
    print(f"    3개 보유 없이 연쇄로 은상자    {pct(sum(chain_only), n)}")
    print(f"    3개 보유 중 한 수 합성 가능    원래 규칙 {pct(sum(raw3), n)} · 지금 마스크 {pct(sum(mask3), n)} · 직후 개봉까지(우선 후보) {pct(sum(exec3), n)}")
    print(f"    은상자 합성                    {pct(sum(made), n)} (직접 {typ['직접']} · 연쇄 {typ['연쇄']})")
    print(f"    합성 은상자 개봉               {pct(sum(opened), n)}")
    if sum(made):
        mg = [g for g in G if g["silver"]]
        print(f"      합성까지 일수(시작 기준) {dist([g['silver']['rel_day'] for g in mg])} · 합성 행동 종류 {dict(Counter(g['silver']['via'] for g in mg))}")
        via3 = [g for g in mg if g["silver"]["type"] == "직접" and g["silver"]["streak"]]
        if via3:
            sk = [g["silver"]["streak"] for g in via3]
            prep = [g for g in via3 if g["silver"]["streak"].get("raw", 0) == 0]
            how = Counter(("입력 대기에서 시작" if g["silver"]["streak"]["phase"] == 0 else "밤 직전 등에서 시작") + "·"
                          + {"드래그": "드래그로 완성", "밤 시작(아침 해소 포함)": "밤·아침 해소로 완성"}.get(g["silver"]["streak"].get("first_raw_via"), "탭·아이템으로 완성")
                          + ("(같은 날)" if g["silver"]["streak"].get("first_raw_day") == g["silver"]["streak"]["day"] else "(뒷날)") for g in prep)
            print(f"      직접 합성 {len(via3)}판 (마지막 3개 보유 구간 기준): 구간 시작 → 합성까지 행동 수 {dist([g['silver']['step'] - g['silver']['streak']['step'] for g in via3], '{:.0f}')}"
                  f" · 구간 시작 때 이미 한 수 거리 {len(via3) - len(prep)}판 · 준비 이동 필요 {len(prep)}판"
                  + (f" (배치 완성까지 행동 수 {dist([g['silver']['streak']['first_raw'] - g['silver']['streak']['step'] for g in prep if g['silver']['streak']['first_raw'] is not None], '{:.0f}')},"
                     f" 시작 패턴 거리 {dict(sorted(Counter(x['dist'] for x in [g['silver']['streak'] for g in prep]).items()))})" if prep else "")
                  + f" · 처음 3개 확보 뒤 손실을 거친 판 {sum(1 for g in via3 if any(l['after_held3'] for l in g['losses']))}")
            if prep:
                print(f"        준비가 필요했던 {len(prep)}판: " + ", ".join(f"{k_} {v}" for k_, v in how.most_common()))
        ch = [g for g in mg if g["silver"]["type"] == "연쇄"]
        if ch:
            print(f"      연쇄 {len(ch)}판: 그 직전 상태가 '2개 보유 중 한 수 연쇄 가능'으로 잡힌 판 {sum(1 for g in ch if g['silver']['chain2'])}")
    c2 = [g for g in G if g["chain2_states"]]
    print(f"    (참고) 2개 보유 중 드래그 한 번으로 연쇄 은상자가 가능했던 판 {pct(len(c2), n)}"
          + (f" — 그중 연쇄로 은상자 {sum(1 for g in c2 if g['silver'] and g['silver']['type'] == '연쇄')}판, 기회 상태 {sum(len(g['chain2_states']) for g in c2)}개"
             f" (우선 규칙은 동상자 3개 이상일 때만 본다)" if c2 else ""))
    lost = [len(g["losses"]) > 0 for g in G]
    never3 = [not (a or b) for a, b in zip(h3, chain_only)]
    rec3 = sum(1 for g in G if g["losses"] and ((g["held3"] and g["held3"]["step"] > g["losses"][0]["step"]) or g["silver"]))
    print(f"    손실 경험(은상자 전 동상자 감소) {pct(sum(lost), n)} · 끝내 3개 못 모음 {pct(sum(never3), n)}")
    print(f"      손실 O·3개 못 모음 {sum(a and b for a, b in zip(lost, never3))} · 손실 O·3개 모음 {sum(a and not b for a, b in zip(lost, never3))}"
          f" (첫 손실 뒤에 3개 확보·은상자 {rec3}) · 손실 X·3개 못 모음 {sum(not a and b for a, b in zip(lost, never3))} · 손실 X·3개 모음 {sum(not a and not b for a, b in zip(lost, never3))}")

    # 3. 끊긴 지점
    cls = [classify(g) for g in G]
    print(f"\n  [3] 끊긴 지점 (게임 분류, 이후 생존 평균)")
    for k_, v in Counter(cls).most_common():
        print(f"    {k_}: {pct(v, n)} · 이후 생존 {np.mean([life[i] for i in range(n) if cls[i] == k_]):.1f}일")
    n3 = [i for i in range(n) if cls[i] == "3개 전에 동상자 잃음"]
    if n3:
        first = Counter(f"{G[i]['losses'][0]['cause']}({G[i]['losses'][0]['b']}→{G[i]['losses'][0]['b_after']})" for i in n3)
        print(f"    · 3개 전에 잃은 {len(n3)}판의 첫 손실: " + ", ".join(f"{k_} {v}" for k_, v in first.most_common()))
        print(f"      끝날 때 보유 동상자: {dict(sorted(Counter(G[i]['end_b'] for i in n3).items()))}")
    nd = [i for i in range(n) if cls[i] == "3개 전에 손실 없이 사망"]
    if nd:
        print(f"    · 손실 없이 사망 {len(nd)}판의 끝 보유: {dict(sorted(Counter(G[i]['end_b'] for i in nd).items()))} · 최대 보유 {dict(sorted(Counter(G[i]['max_b'] for i in nd).items()))}")
    w3 = [i for i in range(n) if G[i]["held3"] is not None and G[i]["silver"] is None]
    if w3:
        end = Counter()
        for i in w3:
            after = [l for l in G[i]["losses"] if l["after_held3"]]
            end[f"잃음({after[0]['cause']})" if after else f"{G[i]['end_b']}개 가진 채 사망"] += 1
        print(f"    · 3개를 모았지만 은상자 못 만든 {len(w3)}판: 3개 뒤 결말 " + ", ".join(f"{k_} {v}" for k_, v in end.most_common()))
        print(f"      3개 보유 상태 수 판당 {np.mean([G[i]['hold3_states'] for i in w3]):.1f} · 패턴 거리(상태 비율) 교환 1번 "
              f"{sum(G[i]['dist3'][1] for i in w3) / max(1, sum(G[i]['hold3_states'] for i in w3)):.0%} · 2번 "
              f"{sum(G[i]['dist3'][2] for i in w3) / max(1, sum(G[i]['hold3_states'] for i in w3)):.0%} · 3번+ "
              f"{sum(G[i]['dist3'][3] for i in w3) / max(1, sum(G[i]['hold3_states'] for i in w3)):.0%}")
    if w3:
        print(f"      교환 두 번 거리(패턴)·스왑 2개 이상 남은 입력 대기 상태가 있던 판 {sum(1 for i in w3 if G[i]['d2_ok'])} / {len(w3)} (상태 판당 {np.mean([G[i]['d2_ok'] for i in w3]):.1f})")
    dn = Counter()
    for g in G:
        dn.update(g["dusk_next"])
    if dn:
        print(f"    · 밤 직전(스왑 0)에 한 수 배치가 있던 {sum(1 for g in G if g['dusk_next'])}판의 다음 결과(횟수): " + ", ".join(f"{k_} {v}" for k_, v in dn.most_common()))
    bl = Counter()
    for g in G:
        bl.update(g["blocked"])
    bg = Counter(k_ for g in G for k_ in g["blocked"])
    if bl:
        print(f"    · 한 수 배치가 있는데 우선 규칙이 못 둔 상태(원인: 상태 수/게임 수): " + ", ".join(f"{k_} {v}/{bg[k_]}" for k_, v in bl.most_common()))
    print(f"    · 합성 실행 가능(우선 후보 있음)했는데 은상자 안 된 판: {sum(1 for g in G if g['exec3'] is not None and g['silver'] is None)}")

    # 처음 3개 확보 시점
    f3 = [g["held3"] for g in G if g["held3"]]
    if f3:
        print(f"\n  처음 3개 확보 시점 ({len(f3)}판): 일차 {dist([x['day'] for x in f3])} · 시작 뒤 {dist([x['rel_day'] for x in f3])}일 · "
              f"하트 {dist([x['hearts'] for x in f3])} · 남은 스왑 {dist([x['swaps'] for x in f3])}")
        print(f"    단계 {dict(Counter('입력 대기' if x['phase'] == 0 else '밤 직전' if x['phase'] == 25 else x['phase'] for x in f3))} · "
              f"패턴 거리 {dict(sorted(Counter(x['dist'] for x in f3).items()))} (1 = 교환 한 번, 3 = 두 번 초과) · "
              f"엔진 한 수 합성(원래 규칙) {sum(1 for x in f3 if x.get('raw', 0) > 0)} · 직후 개봉까지 {sum(1 for x in f3 if x.get('open', 0) > 0)}")
    # 은상자 전 손실 시점
    L = [l for g in G for l in g["losses"]]
    if L:
        print(f"\n  은상자 전 동상자 손실 {len(L)}건 ({sum(lost)}판): 원인 " + ", ".join(f"{k_} {v}" for k_, v in Counter(l["cause"] for l in L).most_common()))
        print(f"    보유 수 변화: " + ", ".join(f"{k_} {v}" for k_, v in Counter(f"{l['b']}→{l['b_after']}" for l in L).most_common()))
        for cause in [k_ for k_, _ in Counter(l["cause"] for l in L).most_common()]:
            x = [l for l in L if l["cause"] == cause]
            print(f"    {cause} {len(x)}건: 일차 {dist([l['day'] for l in x])} · 하트 {dist([l['hearts'] for l in x])} · 스왑 {dist([l['swaps'] for l in x])} · "
                  f"3개 보유 뒤 {sum(l['after_held3'] for l in x)}건")
    return cls


def pick(G, cls, fin, start, want):
    """리플레이 사례 고르기: 조건에 맞는 판 중 이후 생존이 중앙값에 가까운 판"""
    cand = [i for i in range(len(G)) if want(G[i], cls[i])]
    if not cand:
        return None
    life = np.array([fin[i] - start[i] for i in cand])
    return cand[int(np.argmin(np.abs(life - np.median(life))))]


def moves_bronze(f):
    """프레임 f의 행동이 동상자를 움직이는 드래그인지 (보드 문자열 기준)"""
    a = f["a"]
    if a >= 168 or f["p"] != 0:
        return False
    c = a // 4
    x, y = c % 6 + 1, c // 6 + 1
    dx, dy = DIRS[a % 4]
    tx, ty = x + dx, y + dy
    dst = f["b"][(ty - 1) * 6 + tx - 1] if 1 <= tx <= 6 and 1 <= ty <= 7 else ""
    return "h2" in (f["b"][(y - 1) * 6 + x - 1], dst)


def marks_of(g, frames):
    final = len(frames)
    m = []
    if g["held2"] is not None:
        m.append((g["held2"]["step"], "동상자 2개 동시 보유"))
    for l in g["losses"]:
        m.append((l["step"], f"동상자 잃음: {l['cause']} ({l['b']}→{l['b_after']}개, 하트 {l['hearts']}·스왑 {l['swaps']})"))
    if g["held3"] is not None:
        h = g["held3"]
        m.append((h["step"], f"처음 3개 동시 보유 (하트 {h['hearts']}·스왑 {h['swaps']}, 교환 거리 {h['dist'] if h['dist'] < 3 else '3+'})"))
    sk = g["silver"]["streak"] if g["silver"] else None
    if sk and sk["step"] != g["held3"]["step"]:
        m.append((sk["step"], f"다시 3개 동시 보유 (하트 {sk['hearts']}·스왑 {sk['swaps']}, 교환 거리 {sk['dist'] if sk['dist'] < 3 else '3+'})"))
    if sk and sk["first_raw"] is not None and sk.get("first_raw_via") == "드래그" and sk["first_raw"] > sk["step"]:
        m.append((sk["first_raw"] - 1, "준비 이동: 이 수로 한 수 합성 배치 완성"))
        m += [(k, "준비 이동: 동상자를 움직인 드래그") for k in range(sk["step"], sk["first_raw"] - 1) if moves_bronze(frames[k])]
    elif (sk["first_raw"] if sk else g["raw3"]) is not None:
        m.append((sk["first_raw"] if sk else g["raw3"], "한 수 합성 배치가 된 상태"))
    for st, a0, a1 in g["forced"]:
        m.append((st, "우선 규칙이 합성 수로 바꿈"))
    if g["silver"] is not None:
        m.append((g["silver"]["step"], f"은상자 합성({g['silver']['type']})"))
    if g["opened"] is not None:
        m.append((g["opened"]["step"], "합성 은상자 개봉"))
    m.append((final - 1, "마지막 행동"))
    seen, out = set(), []
    for st, t in sorted(m):
        if (st, t) not in seen:
            seen.add((st, t))
            out.append([int(st), t])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--normal-games", type=int, default=1024)
    p.add_argument("--normal-seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--skip", default="", help="건너뛸 조건: stage3,normal")
    p.add_argument("--out", default="", help="게임별 기록 JSON 접두사 (<접두사>_stage3.json, _normal.json)")
    p.add_argument("--replay", default="", help="대표 리플레이 기록(build_replay.py 입력) 경로")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6)}
    pol = Policy(args.ckpt, dev)
    res = {}
    if "stage3" not in args.skip:
        t0 = time.time()
        hold, k, _ = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
        starts = (hold, k, args.pool_seed)
        fin, fins, c = run(pol, args.games, args.seed, rule, starts)
        cls = report(f"{args.ckpt} · 3단계(동상자 2개) {args.games}판 · 한 수 은상자 합성 우선 · {time.time() - t0:.0f}s", fin, fins, c, args.games, True)
        res["stage3"] = (fin, fins, c, cls, starts, args.games, args.seed)
    if "normal" not in args.skip:
        t0 = time.time()
        fin, fins, c = run(pol, args.normal_games, args.normal_seed, rule)
        cls = report(f"{args.ckpt} · 정상 시작 {args.normal_games}판 (시드 {args.normal_seed}) · 한 수 은상자 합성 우선 · {time.time() - t0:.0f}s",
                     fin, fins, c, args.normal_games, False)
        res["normal"] = (fin, fins, c, cls, None, args.normal_games, args.normal_seed)
    if args.out:
        for name, (fin, fins, c, cls, _, n, _) in res.items():
            games = [{"game": i, "start_day": int(c.start[i]), "final": int(fin[i]), "class": cls[i],
                      **{k_: (dict(v) if isinstance(v, Counter) else v) for k_, v in c.g[i].items()}} for i in range(n)]
            with open(f"{args.out}_{name}.json", "w") as f:
                json.dump(games, f, ensure_ascii=False, default=int)
    if args.replay:
        record(args, pol, rule, res)


def record(args, pol, rule, res):
    """대표 사례 3판을 골라 같은 조건으로 다시 두며 프레임을 기록한다(결과가 처음과 같은지 확인)"""
    def prep_ok(g, c, min_moves):
        s_ = g["silver"]
        if not (c.endswith("개봉") and s_["type"] == "직접" and s_["streak"] and s_["streak"].get("raw", 0) == 0):
            return False
        k_ = s_["streak"]
        return (k_["phase"] == 0 and k_["first_raw"] is not None and k_["first_raw"] - k_["step"] >= min_moves
                and k_.get("first_raw_via") == "드래그")

    wants = [
        ("stage3", "두 개를 유지하지 못한 사례",
         lambda g, c: c == "3개 전에 동상자 잃음" and g["losses"][0]["b"] == 2 and g["losses"][0]["cause"] == "비상 개봉"),
        ("stage3", "세 개가 있지만 한 수 배치를 못 한 사례", lambda g, c: c == "3개 모았지만 한 수 배치 못 만듦" and g["held3"]["step"] > 0),
        ("normal", "준비 이동을 거쳐 은상자까지 성공한 사례 (정상 시작)", lambda g, c: prep_ok(g, c, 1)),
        ("stage3", "준비 이동을 거쳐 은상자까지 성공한 사례", lambda g, c: prep_ok(g, c, 2)),
    ]
    picks = {}
    for name, label, want in wants:
        if name not in res or any(l.startswith(label.split(" (")[0]) for _, l in picks.values()):
            continue
        fin, fins, c, cls, *_ = res[name]
        i = pick(c.g, cls, fin, c.start, want)
        if i is not None:
            picks[(name, i)] = (name, label)
    games = []
    for name in ("stage3", "normal"):
        sel = [i for (nm, i) in picks if nm == name]
        if not sel:
            continue
        fin, fins, c, cls, starts, n, seed = res[name]
        fr = [[] if i in sel else None for i in range(n)]
        fin2, _, c2 = run(pol, n, seed, rule, starts, frames=fr)
        assert (fin2 == fin).all(), "다시 둔 결과가 처음과 다르다"
        for i in sel:
            g = c.g[i]
            frames = mark_off(fr[i])
            for st, a0, a1 in g["forced"]:
                frames[st]["base"] = a0
            marks = marks_of(g, frames)
            label = picks[(name, i)][1]
            where = "3단계 연습(동상자 2개 배치)" if name == "stage3" else "정상 시작"
            note = f"{label} · {where} 게임 {i} · {c.start[i]}일 시작, {fin[i]}일 사망 · 분류: {cls[i]}"
            games.append({"id": f"{name}-{i}", "note": note, "search": {"day": int(fin[i]), "frames": frames}, "marks": marks,
                          "label": label})
            print(f"{note}\n  " + "\n  ".join(f"행동 {st + 1}: {t}" for st, t in marks))
    ui = {"title": "동상자 → 은상자 흐름 리플레이", "single": True, "flip": "우선 규칙이 바꾼 수",
          "sub": f"{args.ckpt}(추가 학습 없음)에 한 수 은상자 합성 우선을 켠 기록입니다. 규칙: 은상자부터 개봉, 동상자는 비상 예외(스왑 0·하트 5 이하·열 수 있는 은·금 없음)만, "
                 "은상자 연 날 기본→동 무기 합성 우선(20,6), 휴식일 버리기 금지. 3단계는 학습에 쓰지 않은 평가 풀에 동상자 2개를 놓은 상태, 정상 시작은 시드 777입니다. "
                 "'주요 행동' 버튼으로 동상자 보유·손실·합성 시점으로 갑니다. 행동 번호는 화면의 '행동 k / N'과 같습니다.",
          "a": "cn7 10M + 한 수 합성 우선", "aShort": "cn7"}
    with open(args.replay, "w") as f:
        json.dump({"ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
