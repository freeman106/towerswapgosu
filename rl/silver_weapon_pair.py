"""동→은 무기 합성 뒤 현재 교사의 후속 운영 짝 비교 (추가 학습·보상·교사 설정 변경 없음).
기준 = 정상 시작 25.0일 교사 실행(teacher.py, 원본 cn7_nf 20M, 1,024판, 시드 12345)과 원본 모델을 그대로 쓴다.
1. 상태 확보: 교사 실행의 결정 기록을 재생한다(distill_data.Replay와 같은 방식: 기록된 교체·두 수 완성만 두고 교사 굴림은 다시 하지 않는다.
   확인: 기록된 결정 보드, 게임별 최종 일차). 적격 상태 = 은상자(3등급)를 이미 연 당일, 입력 대기, 휴식일 아님, 남은 스왑 ≥ --min-swaps,
   드래그 한 번으로 같은 종류 동 공격 무기 3개(드래그 전 보드에 있던 것)를 은 공격 무기로 합치는 후보가 있음
   (VecEnv.silver_weapon_plans: 합성 우선 규칙만 끈 복제본에서 원래 게임 규칙상 유효, 가상 판정 시드 4개 모두에서 같은 결과·금 없음).
   교사 두 수 계획의 완성 수 차례인 상태는 그 완성을 A가 이어받을 수 없어 다음 상태로 미룬다(미룬 수를 보고).
   게임마다 첫 적격 상태 하나, 게임 번호 순으로 --states개(이후 결과를 보지 않는다).
2. 상태마다 가상 진행 난수 4개(교사 판정용 가상 시드·교사 굴림 시드와 별개)로 복제해 두 조건을 실제 게임 끝까지 둔다.
   같은 복제 번호 = 같은 진행 난수·같은 샘플링 잡음·같은 교사 굴림 시드(두 조건을 따로 실행하되 슬롯 배치를 같게 둔다).
   A: 현재 교사 그대로. B: 첫 행동에 동→은 합성 한 수(후보끼리 정책 확률 재정규화, 원래 실행의 같은 샘플링 잡음으로 argmax;
   기본→동 합성 우선 규칙이 가리면 그 한 수만 VecEnv.allow_once로 우회), 이후 A와 같은 교사·규칙.
   A가 첫 행동에서 스스로 후보를 두면 그 복제는 개입이 아니다(B = A로 계산).
3. 기록: 개입 직전, 그날 밤(0), 이후 1~5밤, 최종 종료. 짝 차이는 상태별로 복제 4개를 먼저 평균하고 그 상태 평균들로 전체 차이·표준오차.
사용: python rl/silver_weapon_pair.py [--states 16] [--out runs/silver_weapon_pair]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from diag_bronze_flow import run
from eval_restrict import CONDS, Restrict
from force_silver_prep import SEEDS
from ppo import A, C, H, S as NS, W, parse_emergency
from record_dayoff_games import mark_off
from search_eval import Policy, play
from teacher import Teacher, adesc

COPIES = 4  # 상태마다 가상 진행 난수 수
NIGHTS = 6  # 그날 밤(0)과 이후 1~5밤
KINDS = "tbc"
KNAME = {"t": "화살탑", "b": "발리스타", "c": "대포"}
TNAME = ("기본", "동", "은", "금")
CHEST = {"1": "일반", "2": "동", "3": "은", "4": "금"}


def dayoff(env, i, day):
    return env.made(i)[2] == day


def strict_ok(p):
    """드래그 전 보드의 같은 종류 동 공격 무기 3개 이상이 사라지거나 등급이 바뀌어 은 공격 무기가 된 수"""
    _, _, _, created, consumed = p
    silv = [c for c in created if c[1] == 3]
    return bool(silv) and all(sum(1 for q in consumed if q[0] == s[0] and q[1] == 2) >= 3 for s in silv)


class Capture(Restrict):
    """교사 실행 재생(기록된 교체·두 수 완성만, 교사 굴림 없음) + 첫 적격 상태 수집"""

    def __init__(self, n, dec, pol, min_swaps, cont_seed):
        super().__init__(n)
        self.rep = {(d["game"], d["step"]): d for d in dec if d["chosen"]}
        self.pending = [None] * n
        self.pstep = 0
        self.pol, self.min_swaps, self.cont_seed = pol, min_swaps, cont_seed
        self.open_day = np.full(n, -1)
        self.gold_open = np.zeros(n, int)
        self.found = {}
        self.just = []
        self.funnel = [set() for _ in range(n)]
        self.deferred = Counter()
        self.board_ok = 0
        self.hold = ts.SearchEnv(COPIES * n)
        self.og = np.zeros((n, C, H, W), np.float32)
        self.os = np.zeros((n, NS), np.float32)
        self.om = np.zeros((n, A), bool)

    def after(self, env, i, a):
        i = int(i)
        c = self.cur[i]
        if c["phase"] in (0, 25) and 168 <= a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            cell = env.board(i)[0][y - 1][x - 1] if y >= 1 else ""
            if cell == "h3":
                self.open_day[i] = c["day"]
                self.funnel[i].add("은상자 개봉")
            elif cell == "h4":
                self.gold_open[i] += 1
        super().after(env, i, a)

    def capture(self, env, live, noise, step):
        cand = []
        for i in live:
            c = self.cur[i]
            if i in self.found or c["phase"] != 0 or c["day"] != self.open_day[i] or dayoff(env, i, c["day"]):
                continue
            self.funnel[i].add("개봉 당일 입력 대기")
            if c["swaps"] < self.min_swaps:
                continue
            self.funnel[i].add(f"스왑 ≥ {self.min_swaps}")
            flat = [x for row in env.board(i)[0] for x in row]
            if not any(flat.count(k + "2") >= 3 for k in KINDS):
                continue
            self.funnel[i].add("같은 종류 동 공격 무기 3개+")
            cand.append(i)
        if not cand:
            return
        P = env.silver_weapon_plans(np.array(cand, np.int64), SEEDS)
        obs = False
        for i, L in zip(cand, P):
            if L:
                self.funnel[i].add("한 수 은 공격 무기 합성(연쇄 포함)")
            strict = [p for p in L if strict_ok(p)]
            if not strict:
                continue
            self.funnel[i].add("동 3개 → 은 한 수")
            if self.pending[i] is not None:
                self.deferred["교사 두 수 계획 완성 차례"] += 1
                continue
            if not obs:
                env.observe(self.og, self.os, self.om)
                obs = True
            lo, _ = self.pol(self.og[i:i + 1], self.os[i:i + 1], np.ones((1, A), bool))  # 합성 우선 규칙이 가린 후보도 원래 로짓으로
            acts = np.array([p[0] for p in strict], np.int64)
            pick = int(acts[np.argmax(lo[0, acts] + noise[i, acts])])
            slot = COPIES * len(self.found)
            seeds = np.random.default_rng([self.cont_seed, i, 31]).integers(1, 2 ** 62, size=COPIES, dtype=np.uint64)
            self.hold.store(env, np.full(COPIES, i, np.int64), np.arange(slot, slot + COPIES, dtype=np.int64), seeds)
            c = self.cur[i]
            cells, tur = env.board(i)
            e = env.chest_info(i)
            self.found[i] = {"game": i, "step": int(self.step[i]), "pstep": step, "day": c["day"], "hearts": c["hearts"], "swaps": c["swaps"],
                             "slot": slot, "board": [x for row in cells for x in row], "turrets": tur, "chests": list(e[0]),
                             "cands": [[int(p[0]), bool(p[1]), list(p[2]), [list(x) for x in p[3]], [list(x) for x in p[4]]] for p in strict],
                             "all_plans": len(L), "pick": pick, "pick_desc": adesc(cells, pick),
                             "rule_hidden": not any(p[1] for p in strict), "pick_in_mask": bool(self.om[i, pick]),
                             "p_pick": float(np.exp(lo[0, pick] - np.logaddexp.reduce(lo[0, self.om[i]]))) if self.om[i, pick] else None}
            self.just.append(i)

    def force(self, env, alive, logits, noise, act):
        forced = Restrict.force(self, env, alive, logits, noise, act)  # 상자 합성 계획(원래 실행과 같음)
        step = self.pstep
        self.pstep += 1
        live = [int(i) for i in np.nonzero(alive)[0]]
        self.just = []
        self.capture(env, live, noise, step)
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
        for i in live:
            if i in forced or i in recheck or self.pending[i] is not None:
                continue
            d = self.rep.get((i, int(self.step[i])))
            if d is None:
                continue
            assert [x for row in env.board(i)[0] for x in row] == d["board"], f"게임 {i} 행동 {self.step[i]}: 보드가 기록과 다름"
            self.board_ok += 1
            a1, a2 = d["chosen"]
            if a1 != act[i]:
                forced[i] = a1
            if a2 >= 0:
                self.pending[i] = {"a2": a2, "done": d.get("완성"), "day": self.cur[i]["day"]}
        for i in self.just:
            self.found[i]["a_orig"] = int(forced.get(i, act[i]))
            self.found[i]["a_orig_desc"] = adesc(env.board(i)[0], self.found[i]["a_orig"])
        return forced


def snap(env, i):
    """공격 무기 [종류 3][등급 4][행 0..7] (0 = 성 포탑), 얼음벽, 하트, 스왑, 누적 무기 생성 [종류 3][등급 4], 누적 상자 생성 [등급 0..5]"""
    cells, tur = env.board(i)
    lay = np.zeros((3, 4, 8), int)
    for r, row in enumerate(cells, start=1):
        for x in row:
            if x[:1] in KINDS:
                lay[KINDS.index(x[0]), min(int(x[1]), 4) - 1, r] += 1
    for x in tur:
        if x[:1] in KINDS and x[1:2].isdigit():
            lay[KINDS.index(x[0]), min(int(x[1]), 4) - 1, 0] += 1
    day, hearts, swaps, _, _ = env.state(i)
    return {"day": day, "hearts": hearts, "swaps": swaps, "lay": lay, "walls": sum(x[:1] == "w" for row in cells for x in row),
            "made": np.array(env.made(i)[0])[:3], "chest": np.array(env.chest_info(i)[1])}


class Cont(Teacher):
    """적격 상태에서 이어 두는 현재 교사(그대로) + 기록. plan(B 실행만): 슬롯 → (개입 드래그, A가 스스로 후보를 뒀는지)"""

    def __init__(self, n, pol, seed, plan=None, frames=None, window=NIGHTS - 1):
        super().__init__(n, pol, seed, True)
        self.ivplan, self.frames, self.window = plan, frames, window  # self.plan은 상자 합성 계획(Ext)이 쓴다
        self.kept = [None] * n
        self.iv = np.zeros(n, bool)
        self.a0 = np.full(n, -1)
        self.R = [{"d0": None, "pre": None, "post": None, "dusk": {}, "swaps": Counter(), "opens": []} for _ in range(n)]
        self.cut = Counter()
        self.cut_game = np.zeros(n, int)

    def force(self, env, alive, logits, noise, act):
        if self.pstep == 0 and self.ivplan:
            iv = {j: a for j, (a, own) in self.ivplan.items() if not own and alive[j]}
            al = alive.copy()
            al[list(iv)] = False
            forced = super().force(env, al, logits, noise, act)
            for j, a in iv.items():
                env.allow_once(j, a)
                forced[j] = a
                self.iv[j] = True
            return forced
        return super().force(env, alive, logits, noise, act)

    def decide_chunk(self, env, jobs, step, forced, logits):
        before = {i: (forced.get(i), self.pending[i]) for i, _, _ in jobs}
        super().decide_chunk(env, jobs, step, forced, logits)
        pos = 0
        for i, _, cands in jobs:
            k = (1 + len(cands)) * self.S
            cut = int(self.sa[pos:pos + k].sum())  # 400행동 상한에 걸려 밤 전에 멈춘 굴림
            self.cut["교사 굴림 슬롯"] += k
            self.cut["상한에 잘린 굴림 슬롯"] += cut
            self.cut["평가한 결정"] += 1
            if cut:
                self.cut["잘린 굴림이 있는 결정"] += 1
                self.cut_game[i] += 1
                if (forced.get(i), self.pending[i]) != before[i]:
                    self.cut["잘린 굴림이 있는데 교체한 결정"] += 1
            pos += k

    def after(self, env, i, a):
        i = int(i)
        c, r = self.cur[i], self.R[i]
        st = int(self.step[i])
        if st == 0:
            r["d0"], r["pre"], self.a0[i] = c["day"], snap(env, i), a
        elif st == 1:
            r["post"] = snap(env, i)
        if c["phase"] == 0 and a < 168:
            r["swaps"][c["day"]] += 1
        if c["phase"] in (0, 25) and 168 <= a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            cell = env.board(i)[0][y - 1][x - 1] if y >= 1 else ""
            if cell[:1] == "h":
                r["opens"].append((c["day"], int(cell[1])))
        if a == 224:
            r["dusk"][c["day"]] = snap(env, i)
        if self.frames is not None and self.frames[i] is not None and c["day"] > r["d0"] + self.window:
            self.kept[i], self.frames[i] = self.frames[i], None
        super().after(env, i, a)


def cont_run(pol, hold, src, rule, seed, plan=None, frames=None, window=NIGHTS - 1):
    """frames: 프레임을 기록할 슬롯 집합(None이면 기록 안 함), window: 시작 뒤 며칠까지 기록할지"""
    n = len(src)
    env = ts.VecEnv(n, seed=seed, no_toss_day_off=True, **rule)
    env.load_from(hold, src, np.arange(n, dtype=np.int64))
    fr = [[] if j in frames else None for j in range(n)] if frames is not None else None
    c = Cont(n, pol, seed, plan, fr, window)
    fins = []
    fin = play(pol, n, seed, False, 0, 0, 0, None, env=env, ctl=c, fin_rec=fins, frames=fr)
    c.finish()
    fins.sort(key=lambda f: f["env"])
    if frames:
        c.kept = [k if k is not None else f for k, f in zip(c.kept, fr)]
    return fin, fins, c


def slot_metrics(fin, c, j):
    """슬롯 하나의 지표. 밤 t(0 = 그날 밤): 도달 = 그날 밤 직전까지 살아 있음, 생존 = 그 밤을 넘김"""
    r = c.R[j]
    d0, pre = r["d0"], r["pre"]
    m = {"d0": d0, "final": int(fin[j]), "life": int(fin[j]) - d0}
    prev = pre
    for t in range(NIGHTS):
        d = d0 + t
        m[f"survive{t}"] = float(fin[j] > d)
        du = r["dusk"].get(d)
        if du is None:
            continue
        dmg = c.night[j].get(d, np.nan)
        m[f"dusk_h{t}"] = du["hearts"]
        m[f"dmg{t}"] = dmg
        if fin[j] > d:
            m[f"morn_h{t}"] = du["hearts"] - dmg
        lay = du["lay"]
        m[f"lay{t}"] = lay
        m[f"walls{t}"] = du["walls"]
        dm = du["made"] - prev["made"]
        m[f"made{t}"] = dm  # [종류 3][등급 4] 그날(t = 0이면 개입 직전부터) 생긴 공격 무기
        m[f"swaps{t}"] = r["swaps"].get(d, 0)
        m[f"chest{t}"] = du["chest"] - pre["chest"]
        if t == 0:  # 합성 뒤: B는 개입 한 수를 뺀 나머지(개입 직후 상태부터), A는 같은 상태부터
            m["made0_after"] = du["made"] - (r["post"] if c.iv[j] else pre)["made"]
            m["swaps0_after"] = m["swaps0"] - int(c.iv[j])
        prev = du
    last = r["dusk"][max(r["dusk"])] if r["dusk"] else pre
    m["chest_all"] = last["chest"] - pre["chest"]
    op = [o for o in r["opens"]]
    m["silver_open_after"] = sum(1 for d, t in op if t == 3)
    m["silver_open_5"] = sum(1 for d, t in op if t == 3 and d <= d0 + NIGHTS - 1)
    m["bronze_open_after"] = sum(1 for d, t in op if t == 2)
    m["normal_open_after"] = sum(1 for d, t in op if t == 1)
    so = [d for d, t in op if t == 3]
    m["next_silver_day"] = so[0] - d0 if so else np.nan
    return m


def se_str(x, fmt="{:+.2f}"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "-"
    if len(x) == 1:
        return fmt.format(x.mean())
    return f"{fmt.format(x.mean())} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f}"


def paired(MA, MB, own, key, f=lambda v: v):
    """상태별 복제 짝 차이(B − A) 평균 → 상태 평균들. 값이 없는 복제(한쪽이 그 밤에 도달 못 함)는 뺀다.
    반환: 상태 평균 차이 목록, A·B 상태 평균, 비교 가능한 복제 수, 상태 수"""
    dg, ag, bg, npair = [], [], [], 0
    for g in range(len(MA)):
        d, a_, b_ = [], [], []
        for k in range(COPIES):
            ma = MA[g][k]
            mb = ma if own[g][k] else MB[g][k]
            if key not in ma or key not in mb:
                continue
            va, vb = f(ma[key]), f(mb[key])
            if np.isnan(va) or np.isnan(vb):
                continue
            d.append(vb - va)
            a_.append(va)
            b_.append(vb)
        if d:
            dg.append(np.mean(d))
            ag.append(np.mean(a_))
            bg.append(np.mean(b_))
            npair += len(d)
    return np.array(dg), np.array(ag), np.array(bg), npair, len(dg)


def line(name, MA, MB, own, key, f=lambda v: v, all_games=True):
    dg, ag, bg, npair, ng = paired(MA, MB, own, key, f)
    tag = "" if all_games else f" · 비교 가능 복제 {npair} / 상태 {ng}"
    print(f"    {name:<30s} A {np.mean(ag) if len(ag) else np.nan:7.2f} · B {np.mean(bg) if len(bg) else np.nan:7.2f} · B − A {se_str(dg)}{tag}")
    return dg


def report(states, MA, MB, own, iv_check, costs, ca, cb, fa, fb, fins_a, fins_b):
    G = len(states)
    print(f"\n=== 동→은 합성 짝 비교: {G}상태 × 가상 진행 난수 {COPIES}개 (A 교사 그대로 / B 동→은 한 수 뒤 교사) ===")
    print("  [개입 직전] 상태 요약")
    for s in states:
        lay = np.zeros((3, 4), int)
        for x in s["board"]:
            if x[:1] in KINDS:
                lay[KINDS.index(x[0]), min(int(x[1]), 4) - 1] += 1
        own_k = sum(own[states.index(s)])
        silv = [c for c in s["pick_cand"][3] if c[1] == 3]
        cons = [q for q in s["pick_cand"][4] if q[1] == 2]
        print(f"    게임 {s['game']:>4d} · {s['day']:>2d}일 · 하트 {s['hearts']:>2d} · 스왑 {s['swaps']:>2d} · 공격 무기 기본/동/은 {lay[:, 0].sum()}/{lay[:, 1].sum()}/{lay[:, 2].sum()}"
              f" · 후보 {len(s['cands'])}개(합성 우선 규칙에 가림: {'예' if s['rule_hidden'] else '아니오'}) · 개입 {s['pick_desc']} (정책 확률 {'규칙에 가림' if s['p_pick'] is None else f"{s['p_pick']:.1e}"})"
              f" → 은 {' '.join(f'{KNAME[c[0]]}({c[2]},{c[3]})' for c in silv)} · 사라진 동 {' '.join(f'({q[2]},{q[3]})' for q in cons)}"
              f" · 원래 실행의 수 {s['a_orig_desc']} · A가 스스로 합성 {own_k}/{COPIES}")
    hidden = sum(s["rule_hidden"] for s in states)
    print(f"  합성 우선 규칙(기본→동)이 모든 동→은 후보를 가린 상태 {hidden}/{G} (그 상태의 개입 한 수만 allow_once로 우회)")
    print(f"  B 개입 한 수의 실제 결과(진행 난수별): {dict(iv_check)}")

    print("\n  [생존] 전체 복제(중간 사망 포함)")
    line("최종 생존일(day)", MA, MB, own, "final")
    line("시작 뒤 생존일", MA, MB, own, "life")
    print(f"    시작 일차 평균 {np.mean([s['day'] for s in states]):.2f}")
    for t in range(NIGHTS):
        line(f"{t}밤 생존율" + (" (그날 밤)" if t == 0 else ""), MA, MB, own, f"survive{t}")
    print("\n  [밤] 하트·피해 (그 밤 직전까지 두 조건 모두 살아 있는 복제만)")
    for t in range(NIGHTS):
        line(f"{t}밤 직전 하트", MA, MB, own, f"dusk_h{t}", all_games=False)
        line(f"{t}밤 피해", MA, MB, own, f"dmg{t}", all_games=False)
    print("\n  [배치] 밤 직전 공격 무기 (두 조건 모두 그 밤 직전에 살아 있는 복제만)")
    for t in (0, 1, 3, 5):
        for ti in range(3):
            line(f"{t}밤 {TNAME[ti]} 공격 무기 수", MA, MB, own, f"lay{t}", lambda v, ti=ti: v[:, ti].sum(), all_games=False)
        for k in range(3):
            line(f"{t}밤 {KNAME[KINDS[k]]} 수", MA, MB, own, f"lay{t}", lambda v, k=k: v[k].sum(), all_games=False)
        for rows, name in (((0,), "성 포탑"), ((1, 2), "1~2행"), ((3, 4), "3~4행"), ((5, 6), "5~6행")):
            line(f"{t}밤 {name} 공격 무기", MA, MB, own, f"lay{t}", lambda v, rows=rows: v[:, :, list(rows)].sum(), all_games=False)
            line(f"{t}밤 {name} 동+ 공격 무기", MA, MB, own, f"lay{t}", lambda v, rows=rows: v[:, 1:, list(rows)].sum(), all_games=False)
        line(f"{t}밤 얼음벽", MA, MB, own, f"walls{t}", all_games=False)
    print("\n  [생산] 그날 공격 무기 생성 (0 = 개입 직전부터 그날 밤까지, B의 0일은 개입 한 수 포함; 두 조건 모두 그날을 둔 복제만)")
    line("0일 합성 뒤 생산(기본)", MA, MB, own, "made0_after", lambda v: v[:, 0].sum())
    line("0일 합성 뒤 동 합성", MA, MB, own, "made0_after", lambda v: v[:, 1].sum())
    line("0일 합성 뒤 은 합성(추가)", MA, MB, own, "made0_after", lambda v: v[:, 2].sum())
    line("0일 합성 뒤 사용 스왑", MA, MB, own, "swaps0_after")
    for t in range(NIGHTS):
        line(f"{t}일 생산(기본)", MA, MB, own, f"made{t}", lambda v: v[:, 0].sum(), all_games=False)
        line(f"{t}일 동 합성", MA, MB, own, f"made{t}", lambda v: v[:, 1].sum(), all_games=False)
        line(f"{t}일 은 합성", MA, MB, own, f"made{t}", lambda v: v[:, 2].sum(), all_games=False)
        line(f"{t}일 사용 스왑(드래그)", MA, MB, own, f"swaps{t}", all_games=False)
    print("\n  [상자 수입] 시작 뒤 (전체 복제)")
    for ti, name in ((1, "일반"), (2, "동"), (3, "은")):
        line(f"{name}상자 생성(끝까지)", MA, MB, own, "chest_all", lambda v, ti=ti: v[ti])
        line(f"{name}상자 생성(5밤까지)", MA, MB, own, f"chest{NIGHTS - 1}", lambda v, ti=ti: v[ti], all_games=False)
    line("은상자 개봉(끝까지)", MA, MB, own, "silver_open_after")
    line("은상자 개봉(5일 안)", MA, MB, own, "silver_open_5")
    line("동상자 개봉(끝까지)", MA, MB, own, "bronze_open_after")
    line("일반상자 개봉(끝까지)", MA, MB, own, "normal_open_after")
    line("다음 은상자 개봉까지 일수(연 복제만)", MA, MB, own, "next_silver_day", all_games=False)

    print("\n  [상태별] 짝 차이(B − A, 복제 4개 평균) — 시작 뒤 생존 · 그날 밤 피해 · 0밤 직전 공격 무기 수 · 0일 생산 · 5~6행 동+ (0밤)")
    for g, s in enumerate(states):
        d = []
        for key, f in (("life", lambda v: v), ("dmg0", lambda v: v), ("lay0", lambda v: v.sum()), ("made0", lambda v: v[:, 0].sum()),
                       ("lay0", lambda v: v[:, 1:, [5, 6]].sum())):
            vals = []
            for k in range(COPIES):
                ma, mb = MA[g][k], (MA[g][k] if own[g][k] else MB[g][k])
                if key in ma and key in mb and not np.isnan(f(ma[key])) and not np.isnan(f(mb[key])):
                    vals.append(f(mb[key]) - f(ma[key]))
            d.append(np.mean(vals) if vals else np.nan)
        lifeA = [MA[g][k]["life"] for k in range(COPIES)]
        lifeB = [(MA[g][k] if own[g][k] else MB[g][k])["life"] for k in range(COPIES)]
        print(f"    게임 {s['game']:>4d}: 생존 A {lifeA} · B {lifeB} → {d[0]:+.2f} · 피해 {d[1]:+.2f} · 공격 무기 {d[2]:+.2f} · 0일 생산 {d[3]:+.2f} · 5~6행 동+ {d[4]:+.2f}")

    print("\n  [확인]")
    for name, c, fins in (("A", ca, fins_a), ("B", cb, fins_b)):
        tr = sum(bool(f["truncated"]) for f in fins)
        print(f"    {name}: 실제 게임 {len(fins)}판 모두 게임 오버로 끝남(안전 상한 200,000행동에 걸린 판 {tr}) · 교사 {dict(c.cut)}"
              f" · 교사 결정 {dict((k, v) for k, v in c.stats.items() if k in ('평가한 결정', '교체', '유지(기존 선택)', '두 수 계획 시작'))}")
    print(f"    실행 비용: {costs}")



def marks_of(fr, forced, first):
    """주요 행동: 첫 행동, 밤 시작·다음 아침 하트, 상자 개봉, 마지막 프레임. 교사·계획이 바꾼 수는 프레임 base로 표시"""
    for st, a0, _ in forced:
        if st < len(fr) and a0 != fr[st]["a"]:
            fr[st]["base"] = int(a0)
    mark_off(fr)
    m = [[0, first]]
    for t, f in enumerate(fr):
        a = f["a"]
        if a == 224:
            m.append([t, f"{f['d']}일 밤 시작 · 하트 {f['h']}"])
            if t + 1 < len(fr):
                g = fr[t + 1]
                m.append([t + 1, f"{g['d']}일 아침 · 하트 {g['h']} (밤 피해 {f['h'] - g['h']})"])
        elif 168 <= a < 216 and f["p"] in (0, 25):
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            cell = f["b"][(y - 1) * 6 + x - 1] if y >= 1 else ""
            if cell[:1] == "h":
                m.append([t, f"{CHEST.get(cell[1], '')}상자 개봉 ({x},{y})"])
    m.append([len(fr) - 1, "기록 마지막 행동"])
    return m


def scenes(states, MA, MB, own, ca, cb):
    """이득·손해 대표 장면: 상태 평균 생존 차이가 가장 큰·작은 상태에서, 개입한 복제 중 차이가 상태 평균에 가장 가까운(같은 부호) 복제"""
    G = len(states)
    diff = [[MB[g][k]["life"] - MA[g][k]["life"] if not own[g][k] else np.nan for k in range(COPIES)] for g in range(G)]
    dg = np.array([np.nanmean(d) if not np.all(np.isnan(d)) else np.nan for d in diff])
    out = []
    for label, g, sign in (("이득", int(np.nanargmax(dg)), 1), ("손해", int(np.nanargmin(dg)), -1)):
        ks = [k for k in range(COPIES) if not own[g][k] and sign * diff[g][k] > 0]
        if not ks or sign * dg[g] <= 0:
            print(f"  대표 장면({label}): 해당 복제 없음")
            continue
        k = min(ks, key=lambda k: abs(diff[g][k] - dg[g]))
        j = g * COPIES + k
        s = states[g]
        silv = [c for c in s["pick_cand"][3] if c[1] == 3]
        ma, mb = MA[g][k], MB[g][k]
        note = (f"{label} 장면 · 게임 {s['game']} {s['day']}일(하트 {s['hearts']}, 스왑 {s['swaps']}) · 가상 진행 난수 {k} · "
                f"개입 {s['pick_desc']} → 은 {' '.join(f'{KNAME[c[0]]}({c[2]},{c[3]})' for c in silv)} · "
                f"최종 A {ma['final']}일 / B {mb['final']}일 (상태 평균 차이 {dg[g]:+.2f}일) · "
                f"그날 밤 피해 A {ma.get('dmg0', np.nan):.0f} / B {mb.get('dmg0', np.nan):.0f}")
        fa_, fb_ = ca.kept[j], cb.kept[j]
        assert fa_ is not None and fb_ is not None, f"슬롯 {j}의 프레임이 기록되지 않았다"
        games_note = f"{label}: 게임 {s['game']} · 난수 {k}"
        out.append({"id": f"{s['game']}-{k}", "label": games_note, "note": note, "diverge": 0,
                    "search": {"day": ma["final"], "frames": fa_, "marks": marks_of(fa_, ca.g[j]["forced"], "적격 상태 · 교사가 그대로 이어 둠")},
                    "base": {"day": mb["final"], "frames": fb_, "marks": marks_of(fb_, cb.g[j]["forced"], f"동→은 합성 개입 {s['pick_desc']}")}})
        print(f"  대표 장면({label}): {note}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dec", default="runs/teacher1024_s12345_decisions.json")
    p.add_argument("--games-json", default="runs/teacher1024_s12345_games.json")
    p.add_argument("--min-swaps", type=int, default=40)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--cont-seed", type=int, default=930)
    p.add_argument("--out", default="")
    p.add_argument("--frames", action="store_true", help="대표 장면용 프레임 기록(모든 슬롯, 시작 뒤 6일)")
    p.add_argument("--scene-slots", default="", help="'게임:복제,...'만 게임 끝까지 프레임 기록(대표 장면 다시 두기). 결과는 --check와 같아야 한다")
    p.add_argument("--check", default="", help="앞선 실행의 결과 JSON: 최종 일차가 같은지 확인")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    dec = json.load(open(args.dec))
    want = np.array(json.load(open(args.games_json))["T"]["final"])
    t0 = time.time()
    cap = Capture(args.games, dec, pol, args.min_swaps, args.cont_seed)
    fin, _, _ = run(pol, args.games, args.seed, rule, None, ctl=cap)
    same = fin == want
    t_rep = time.time() - t0
    print(f"재생 {t_rep:.0f}s · 최종 일차 일치 {same.sum()}/{len(same)} · 결정 보드 일치 {cap.board_ok}", flush=True)
    assert same.all(), f"최종 일차가 다른 게임 {np.nonzero(~same)[0][:10]}"
    keys = ["은상자 개봉", "개봉 당일 입력 대기", f"스왑 ≥ {args.min_swaps}", "같은 종류 동 공격 무기 3개+", "한 수 은 공격 무기 합성(연쇄 포함)", "동 3개 → 은 한 수"]
    print("  적격 깔때기(게임 수, 1,024판): " + " → ".join(f"{k} {sum(k in f for f in cap.funnel)}" for k in keys)
          + f" → 첫 적격 상태 확보 {len(cap.found)} · 미룸 {dict(cap.deferred)} · 금상자 개봉 판 {int((cap.gold_open > 0).sum())}")
    elig = sorted(cap.found)
    chosen = elig[:args.states]
    if len(chosen) < args.states:
        print(f"  목표 {args.states}상태 중 {len(chosen)}상태만 확보 (조건 완화 없음)")
    states = [cap.found[g] for g in chosen]
    for s in states:
        s["pick_cand"] = next(c for c in s["cands"] if c[0] == s["pick"])
    n = COPIES * len(states)
    # 슬롯 j = 상태 g × 4 + 복제 k (두 조건 같은 배치). 저장해 둔 복제(진행 난수 k로 바꾼 적격 상태)를 그대로 불러온다
    src = np.concatenate([np.arange(s["slot"], s["slot"] + COPIES) for s in states]).astype(np.int64)
    costs = {"재생": f"{t_rep:.0f}s"}
    frames, window = None, NIGHTS - 1
    if args.scene_slots:
        gi = {s["game"]: g for g, s in enumerate(states)}
        frames = {gi[int(x.split(":")[0])] * COPIES + int(x.split(":")[1]) for x in args.scene_slots.split(",")}
        window = 10 ** 6
    elif args.frames:
        frames = set(range(n))
    t1 = time.time()
    fa, fins_a, ca = cont_run(pol, cap.hold, src, rule, args.cont_seed, frames=frames, window=window)
    costs["A"] = f"{time.time() - t1:.0f}s"
    print(f"  A 끝: {costs['A']} · MPS {torch.mps.driver_allocated_memory() / 2 ** 30 if torch.backends.mps.is_available() else 0:.1f}GB", flush=True)
    own = [[bool(any(ca.a0[g * COPIES + k] == c[0] for c in s["cands"])) for k in range(COPIES)] for g, s in enumerate(states)]
    plan = {g * COPIES + k: (s["pick"], own[g][k]) for g, s in enumerate(states) for k in range(COPIES)}
    t1 = time.time()
    fb, fins_b, cb = cont_run(pol, cap.hold, src, rule, args.cont_seed, plan=plan, frames=frames, window=window)
    if args.check:
        ref = json.load(open(args.check))
        for name, fin_ in (("A", fa), ("B", fb)):
            want_ = np.array([m["final"] for r in ref[name] for m in r])
            got = np.array([fin_[j] for j in range(n)])
            print(f"  다시 둔 {name} 최종 일차가 앞선 실행과 같음 {int((got == want_).sum())}/{n}", flush=True)
    costs["B"] = f"{time.time() - t1:.0f}s"
    iv_check = Counter()
    for j in range(n):
        if not cb.iv[j]:
            continue
        r = cb.R[j]
        dm = r["post"]["made"] - r["pre"]["made"]
        iv_check["은 공격 무기 생성" if dm[:, 2].sum() > 0 else "은 공격 무기 안 생김"] += 1
    MA = [[slot_metrics(fa, ca, g * COPIES + k) for k in range(COPIES)] for g in range(len(states))]
    MB = [[slot_metrics(fb, cb, g * COPIES + k) for k in range(COPIES)] for g in range(len(states))]
    same_own = [(fa[g * COPIES + k], fb[g * COPIES + k]) for g in range(len(states)) for k in range(COPIES) if own[g][k]]
    if same_own:
        print(f"  A가 스스로 합성한 복제 {len(same_own)}개: B 실행의 같은 슬롯 최종 일차가 A와 같음 {sum(a == b for a, b in same_own)}/{len(same_own)} (분석은 B = A)")
    report(states, MA, MB, own, iv_check, costs, ca, cb, fa, fb, fins_a, fins_b)
    if args.out:
        slim = lambda m: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
        with open(args.out + ".json", "w") as f:
            json.dump({"args": vars(args), "states": states, "own": own, "A": [[slim(m) for m in r] for r in MA],
                       "B": [[slim(m) for m in r] for r in MB], "costs": costs,
                       "cut": {"A": dict(ca.cut), "B": dict(cb.cut)}}, f, ensure_ascii=False, default=int)
        if frames is not None:
            games = scenes(states, MA, MB, own, ca, cb)
            ui = {"title": "동→은 합성 뒤 교사 운영", "flip": "교사·상자 계획이 바꾼 수", "a": "A: 교사 그대로", "b": "B: 동→은 합성 한 수 뒤 교사",
                  "aShort": "A", "bShort": "B",
                  "sub": "정상 시작 교사 실행(cn7_nf 20M + 1~2수 생산·합성 교사, 시드 12345)을 재생해 얻은 실제 상태에서, 은상자를 연 당일 스왑이 40개 이상 남았고 "
                         "드래그 한 번으로 동 공격 무기 3개를 은 무기로 합칠 수 있을 때 두 조건을 같은 가상 진행 난수·샘플링 잡음·교사 굴림 시드로 둔 기록입니다. "
                         "<b>A</b>: 현재 교사가 그대로 진행. <b>B</b>: 첫 행동에 동→은 합성 한 수만 개입하고 이후 같은 교사. " + ("기록은 게임 끝까지입니다." if args.scene_slots else "기록은 시작 뒤 6일(그날 밤과 이후 5밤)까지입니다.")}
            with open(args.out + "_replay.json", "w") as f:
                json.dump({"ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False, default=int)


if __name__ == "__main__":
    main()
