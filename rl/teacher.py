"""1~2수 생산·합성 교사 (추가 학습·보상 변경 없음). 기준 = cn7_nf 20M(대포 방향 금지 학습 +10M)에 기존 상자 개봉 규칙
(은상자부터, 동상자 비상 예외 2,0,5)·무기 합성 우선(20,6)·상자 합성 계획(연쇄 포함 한 수 은상자 합성 + 준비 이동 1수)을 그대로 둔다.
후보(VecEnv.teacher_plans, 실제 난수 대신 가상 시드 4개를 쓰고 모든 시드에서 공격 무기 생성 결과가 같은 수만):
  한 수 생산(자원 매치 → 기본 공격 무기) / 한 수 합성(기본 → 동) / 준비 1수 + 완성 1수(준비 없이 같은 결과가 나는 경로는 제외).
  은 이상 무기가 생기는 수는 후보가 아니다(은 무기 합성 우선 규칙 없음).
비교: 정책이 고른 수(기준) + 교사 후보(첫 수와 결과 무기가 같은 경로는 하나로, 한 수 우선 뒤 정책 확률 순으로 최대 NCAND개)를
  같은 가상 난수 묶음 S개로 복제해 같은 정책·같은 샘플링 잡음으로 남은 하루와 그 밤을 진행한다(두 수 후보는 첫 굴림에서 둘째 수, 무효면 정책).
  기록: 그 밤 사망, 밤 직전 하트, 밤 피해, 다음 아침 하트(사망 = 0), 다음 아침 공격 무기 배치, Q = 그 밤까지 보상/σ + γ^일수·V(다음 아침).
선택 기준: 교사 후보가 (1) 그 밤 사망 수가 기준보다 많지 않고 (2) 다음 아침 하트 평균이 기준보다 낮지 않으며
  (3) Q 짝 차이 평균이 0보다 크고 z(2)·표준오차보다 크면 교체 후보, 여럿이면 Q 차이가 가장 큰 것. 합성 수나 1·3·9·27 전력으로 순위를 정하지 않는다.
두 수 계획은 첫 수 뒤 실제 상태에서 완성 수를 가상 시드로 다시 판정해 유효하면 끝까지 둔다. 상자 합성 계획이 개입한 상태와 휴식일에는 교사가 개입하지 않는다.
사용: python rl/teacher.py [--games 256 --seed 777] [--out 접두사]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from diag_bronze_flow import run
from eval_restrict import CONDS, Restrict, report
from force_silver_prep import SEEDS, se
from ppo import A, C, H, S as NS, W, parse_emergency
from search_eval import DIRS, Policy, act_cat

T_MAX = 400
MAX_SLOTS = 8192  # 한 번에 굴리는 복제 슬롯 수 상한
NAME = {"l": "나무", "s": "돌", "i": "철", "g": "보물", "d": "얼음", "t": "화살탑", "b": "발리스타", "c": "대포", "w": "얼음벽",
        "h": "상자", "a": "모루", "T": "TNT", "F": "요정", "H": "요정집", "I": "빙산", "C": "대포자리"}
TIER = {"0": "", "1": "기본 ", "2": "동 ", "3": "은 ", "4": "금 "}


def tile_name(code):
    if code in ("..", "~~", ""):
        return {"..": "빈칸", "~~": "물", "": "밖"}[code]
    return (TIER.get(code[1], "") if code[0] in "tbcwh" else "") + NAME.get(code[0], code[0])


def adesc(cells, a):
    """행동 설명 (보드 문자열 기준)"""
    if a >= 168:
        return act_cat(cells, 0, a)
    c = a // 4
    x, y = c % 6 + 1, c // 6 + 1
    dx, dy = DIRS[a % 4]
    tx, ty = x + dx, y + dy
    dst = cells[ty - 1][tx - 1] if 1 <= tx <= 6 and 1 <= ty <= 7 else ("성" if ty == 0 else "")
    return f"{tile_name(cells[y - 1][x - 1])}({x},{y}) {'↑↓←→'[a % 4]} {tile_name(dst) if dst != '성' else '성'}"


def swap_key(a):
    """드래그가 바꾸는 두 칸 (같은 교환을 반대쪽에서 끈 드래그는 같은 키)"""
    if a < 0 or a >= 168:
        return a
    c = a // 4
    x, y = c % 6 + 1, c // 6 + 1
    dx, dy = DIRS[a % 4]
    return frozenset({(x, y), (x + dx, y + dy)})


def drag_xy(a):
    c = a // 4
    return f"({c % 6 + 1},{c // 6 + 1}) {'↑↓←→'[a % 4]}"


def plan_kind(p):
    m = np.array(p[2]).reshape(3, 4)
    return ("두 수" if p[1] >= 0 else "한 수") + ("·생산" if m[:, 0].sum() else "") + ("·합성" if m[:, 1].sum() else "")


def layout(cells, turrets):
    """공격 무기 기본/동/은+ 수, 5~6행 공격 무기, 얼음벽"""
    n = [0, 0, 0]
    bottom, walls = 0, 0
    for r, row in enumerate(cells, start=1):
        for x in row:
            if x[:1] in "tbc":
                n[min(int(x[1]), 3) - 1] += 1
                bottom += r in (5, 6)
            walls += x[:1] == "w"
    for x in turrets:
        if x[:1] == "t" and x[1:2].isdigit():
            n[min(int(x[1]), 3) - 1] += 1
    return n + [bottom, walls]


class Teacher(Restrict):
    def __init__(self, n, pol, seed, teach, S=8, ncand=6, z=2.0, scene_keys=()):
        super().__init__(n)
        self.scene_keys = set(scene_keys)  # (게임, 행동 번호): 이 결정에서 기존 선택 줄과 교사 줄을 가상 시드 0으로 굴려 프레임을 남긴다
        self.scenes = []
        self.pol, self.seed, self.teach, self.S, self.ncand, self.z = pol, seed, teach, S, ncand, z
        self.pstep = 0
        self.pending = [None] * n  # (둘째 수, 결과 서명, 결정 기록 번호, 날)
        self.dec = []               # 결정 기록
        self.stats = Counter()
        self.kinds = Counter()
        self.reject = Counter()
        self.wtoss = np.zeros(n, int)
        self.rtrip = np.zeros(n, int)
        self.lastpair = [None] * n
        self.tday = [set() for _ in range(n)]
        self.cap = 0

    def after(self, env, i, a):
        i = int(i)
        c = self.cur[i]
        if c["phase"] == 0 and a < 168:
            cells = env.board(i)[0]
            cc = a // 4
            x, y = cc % 6 + 1, cc // 6 + 1
            dx, dy = DIRS[a % 4]
            toss = act_cat(cells, 0, a) == "버리기"
            if toss and cells[y - 1][x - 1][:1] in "tbcw":
                self.wtoss[i] += 1
            pair = (c["day"], frozenset({(x, y), (x + dx, y + dy)}))
            if not toss and self.lastpair[i] == pair and self.lastcat[i] == "매치 없는 이동":
                self.rtrip[i] += 1
            self.lastpair[i] = None if toss else pair
        super().after(env, i, a)

    def force(self, env, alive, logits, noise, act):
        forced = super().force(env, alive, logits, noise, act)
        step = self.pstep
        self.pstep += 1
        if not self.teach:
            return forced
        live = [int(i) for i in np.nonzero(alive)[0]]
        # 진행 중인 두 수 계획: 실제 상태에서 완성 수를 다시 판정
        recheck = []
        for i in live:
            pend = self.pending[i]
            if pend is None:
                continue
            c = self.cur[i]
            if i in forced or c["day"] != pend[3]:
                self.stats["두 수 계획: 상자 계획 우선·날 바뀜으로 중단"] += 1
                self.pending[i] = None
            elif c["phase"] == 0:
                recheck.append(i)
        if recheck:
            P = env.teacher_plans(np.array(recheck, np.int64), SEEDS)
            for i, L in zip(recheck, P):
                a2, sig, di, _ = self.pending[i]
                self.pending[i] = None
                ok = any(p[0] == a2 and p[1] < 0 and list(p[2]) == sig for p in L)
                self.stats["두 수 계획: 재확인 유효 → 완성 수" if ok else "두 수 계획: 재확인 실패 → 정책"] += 1
                if di is not None:
                    self.dec[di]["완성"] = bool(ok)
                if ok:
                    if act[i] != a2:
                        forced[i] = a2
                        self.stats["두 수 계획: 완성 수 강제"] += 1
                    self.tday[i].add(self.cur[i]["day"])
        cand_games = [i for i in live if i not in forced and i not in recheck and self.pending[i] is None
                      and self.cur[i]["phase"] == 0 and self.cur[i]["swaps"] >= 1
                      and not (self.cur[i]["day"] % 10 == 1 and env.made(i)[2] == self.cur[i]["day"])]
        if not cand_games:
            return forced
        P = env.teacher_plans(np.array(cand_games, np.int64), SEEDS)
        jobs = []
        for i, L in zip(cand_games, P):
            base = int(act[i])
            seen, cands = set(), []
            for p in sorted(L, key=lambda p: (p[1] >= 0, -logits[i, p[0]])):
                key = (swap_key(p[0]), swap_key(p[1]), tuple(sorted(p[3])))
                if key in seen:
                    continue
                seen.add(key)
                if p[1] < 0 and swap_key(p[0]) == swap_key(base):
                    self.stats["정책이 스스로 교사 한 수를 고름"] += 1
                    continue
                cands.append(p)
                if len(cands) >= self.ncand:
                    break
            if cands:
                jobs.append((i, base, cands))
        if not jobs:
            return forced
        self.decide(env, jobs, step, forced, logits)
        if torch.backends.mps.is_available() and step % 10 == 0:
            if step % 200 == 0:
                print(f"  [교사] 스텝 {step} · 진행 중 {len(live)}판 · MPS {torch.mps.driver_allocated_memory() / 2 ** 30:.1f}GB", flush=True)
            torch.mps.empty_cache()  # 굴림마다 배치 크기가 달라 MPS가 크기별 버퍼를 쌓는다(1,024판에서 메모리 초과로 중단)
        return forced

    def decide(self, env, jobs, step, forced, logits):
        """굴림 슬롯이 MAX_SLOTS를 넘지 않게 나눠 판단한다 (가상 시드·잡음은 게임·시드 번호로 정해져 나눠도 같다)"""
        chunk, size = [], 0
        for job in jobs:
            k = (1 + len(job[2])) * self.S
            if chunk and size + k > MAX_SLOTS:
                self.decide_chunk(env, chunk, step, forced, logits)
                chunk, size = [], 0
            chunk.append(job)
            size += k
        if chunk:
            self.decide_chunk(env, chunk, step, forced, logits)

    def decide_chunk(self, env, jobs, step, forced, logits):
        S, pol = self.S, self.pol
        idx, first, second, which, jid = [], [], [], [], []
        for g, (i, base, cands) in enumerate(jobs):
            for a1, a2 in [(base, -1)] + [(p[0], p[1]) for p in cands]:
                for j in range(S):
                    idx.append(i)
                    first.append(a1)
                    second.append(a2)
                    which.append(j)
                    jid.append(g)
        u = len(idx)
        if u > self.cap:
            self.cap = max(u, min(2 * self.cap, MAX_SLOTS + 64))
            self.se = ts.SearchEnv(self.cap)
            self.sg = np.zeros((self.cap, C, H, W), np.float32)
            self.ss = np.zeros((self.cap, NS), np.float32)
            self.sm = np.zeros((self.cap, A), bool)
            self.sa = np.zeros(self.cap, bool)
            self.buf = {k: np.zeros(self.cap, t) for k, t in (("ret", np.float32), ("days", np.float32), ("dead", bool),
                                                               ("lost", np.float32), ("dusk", np.int32), ("hearts", np.int32))}
        idx, first, second, which, jid = (np.array(v, np.int64) for v in (idx, first, second, which, jid))
        n = len(self.start)
        seeds = np.random.default_rng([self.seed, step, 7]).integers(1, 2 ** 62, size=(n, S), dtype=np.uint64)
        se_, sg, ss, sm, sa = self.se, self.sg, self.ss, self.sm, self.sa
        se_.fork(env, idx, first, seeds[idx, which], sg, ss, sm, sa)
        noise = np.random.default_rng([self.seed, step, 9]).gumbel(size=(S, T_MAX, A)).astype(np.float32)
        t = 0
        while sa[:u].any() and t < T_MAX:
            on = np.nonzero(sa[:u])[0]
            lo, _ = pol(sg[on], ss[on], sm[on])
            a_s = np.zeros(len(sa), np.int64)
            a_s[on] = (lo + noise[which[on], t]).argmax(1)
            if t == 0:
                sec = second[on]
                ok = (sec >= 0) & sm[on, np.maximum(sec, 0)]
                a_s[on[ok]] = sec[ok]
            se_.step(a_s, sg, ss, sm, sa)
            t += 1
        b = self.buf
        se_.results(b["ret"], b["days"], b["dead"], b["lost"])
        se_.night(b["dusk"], b["hearts"])
        _, v = pol(sg[:u], ss[:u], sm[:u])
        dead = b["dead"][:u].copy()
        q = b["ret"][:u] / pol.sigma + pol.gamma ** b["days"][:u] * v * (~dead)
        mh = np.where(dead, 0, b["hearts"][:u]).astype(float)
        dusk = b["dusk"][:u].astype(float)
        dmg = np.where(dusk >= 0, np.where(dead, dusk, dusk - b["hearts"][:u]), np.nan)
        pos = 0
        for g, (i, base, cands) in enumerate(jobs):
            nc = 1 + len(cands)
            sl = slice(pos, pos + nc * S)
            Q, MH, DE, DK, DM = (x[sl].reshape(nc, S) for x in (q, mh, dead, dusk, dmg))
            self.stats["평가한 결정"] += 1
            best, best_d = None, 0.0
            rows = []
            for c in range(1, nc):
                d = Q[c] - Q[0]
                mean, sem = d.mean(), d.std(ddof=1) / np.sqrt(S)
                why = ""
                if DE[c].sum() > DE[0].sum():
                    why = "사망 증가"
                elif MH[c].mean() < MH[0].mean():
                    why = "아침 하트 감소"
                elif not (mean > 0 and mean > self.z * sem):
                    why = "Q 이득 불충분"
                rows.append((c, mean, sem, why))
                if not why and mean > best_d:
                    best, best_d = c, mean
            for _, _, _, why in rows:
                self.reject[why or "통과"] += 1
            day = self.cur[i]["day"]
            chosen = cands[best - 1] if best else None
            logit = best is not None or (self.stats["평가한 결정"] % 25 == 0)
            di = None
            if logit:
                cells = self.cur[i]["cells"] or env.board(i)[0]
                lay = [layout(*se_.board(pos + c * S + j)) for c in range(nc) for j in range(S)]
                lay = np.array(lay, float).reshape(nc, S, 5).mean(1)
                cand_rec = []
                for c in range(nc):
                    p = None if c == 0 else cands[c - 1]
                    rec = {"수": [adesc(cells, base)] if c == 0 else [adesc(cells, p[0])] + ([f"(준비 뒤) {drag_xy(p[1])}"] if p[1] >= 0 else []),
                           "종류": "정책 선택" if c == 0 else plan_kind(p), "사망": int(DE[c].sum()),
                           "밤 직전 하트": round(float(np.nanmean(np.where(DK[c] >= 0, DK[c], np.nan))), 2),
                           "밤 피해": round(float(np.nanmean(DM[c])), 2), "아침 하트": round(float(MH[c].mean()), 2),
                           "Q": round(float(Q[c].mean()), 3), "배치(기본/동/은+/5~6행/얼음벽)": [round(float(x), 1) for x in lay[c]]}
                    if c:
                        _, mean, sem, why = rows[c - 1]
                        rec.update({"ΔQ": round(float(mean), 3), "±": round(float(sem), 3), "판정": why or "통과",
                                    "결과 무기": p[3], "사라진 무기": p[4], "자원 칸 변화": round(p[5], 2), "스왑": 2 if p[1] >= 0 else 1})
                    cand_rec.append(rec)
                self.dec.append({"game": i, "step": int(self.step[i]), "day": day, "hearts": self.cur[i]["hearts"], "swaps": self.cur[i]["swaps"],
                                 "board": [x for row in cells for x in row], "base": base, "chosen": [chosen[0], chosen[1]] if chosen else None,
                                 "candidates": cand_rec})
                di = len(self.dec) - 1
            if (i, int(self.step[i])) in self.scene_keys:
                self.scenes.append(self.record_scene(env, i, base, chosen, seeds[i, 0], noise[0], logits[i], di))
            if chosen is None:
                self.stats["유지(기존 선택)"] += 1
                pos += nc * S
                continue
            self.kinds[plan_kind(chosen)] += 1
            self.stats["교체"] += 1
            self.tday[i].add(day)
            if chosen[0] != base:
                forced[i] = int(chosen[0])
            else:
                self.stats["교체: 정책 첫 수 + 교사 완성 수"] += 1
            if chosen[1] >= 0:
                self.pending[i] = (int(chosen[1]), list(chosen[2]), di, day)
                self.stats["두 수 계획 시작"] += 1
            pos += nc * S


def frame(st, board, a, lo):
    day, hearts, swaps, phase, score = st
    cells, tur = board
    pr = np.exp(lo - lo.max())
    pr /= pr.sum()
    top = np.argsort(-pr)[:3]
    return {"d": day, "h": hearts, "s": int(swaps), "p": phase, "sc": int(score), "b": [x for row in cells for x in row], "t": tur,
            "a": int(a), "pa": round(float(pr[a]), 3), "top": [[int(x), round(float(pr[x]), 3)] for x in top if pr[x] > 0.005]}


def record_scene(self, env, i, base, chosen, seed, noise0, logit_i, di):
    """결정 상태에서 기존 선택 줄(0)과 교사 줄(1)을 같은 가상 시드·같은 잡음으로 그날 밤까지 진행한 프레임"""
    pol = self.pol
    first = [base, chosen[0] if chosen else base]
    second = chosen[1] if chosen else -1
    se2 = ts.SearchEnv(2)
    g = np.zeros((2, C, H, W), np.float32)
    s_ = np.zeros((2, NS), np.float32)
    m = np.zeros((2, A), bool)
    a_ = np.zeros(2, bool)
    st0, b0 = env.state(i), env.board(i)
    frames = [[frame(st0, b0, first[0], logit_i)], [frame(st0, b0, first[1], logit_i)]]
    if first[1] != base:
        frames[1][0]["base"] = int(base)
    se2.fork(env, np.array([i, i], np.int64), np.array(first, np.int64), np.array([seed, seed], np.uint64), g, s_, m, a_)
    t = 0
    while a_.any() and t < T_MAX:
        lo, _ = pol(g, s_, m)
        act = (lo + noise0[t]).argmax(1)
        forced2 = t == 0 and second >= 0 and a_[1] and m[1, second]
        if forced2:
            pol_a = int(act[1])
            act[1] = second
        for k in (0, 1):
            if a_[k]:
                frames[k].append(frame(se2.state(k), se2.board(k), act[k], lo[k]))
        if forced2 and pol_a != second:
            frames[1][-1]["base"] = pol_a
        se2.step(act.astype(np.int64), g, s_, m, a_)
        t += 1
    lo, _ = pol(g, s_, m)
    for k in (0, 1):
        st = se2.state(k)
        if st[3] != 11:  # 다음 아침 상태 (게임 오버면 마지막 밤 시작 프레임으로 끝)
            frames[k].append(frame(st, se2.board(k), int((lo[k] + noise0[min(t, T_MAX - 1)]).argmax()), lo[k]))
    return {"game": i, "dec": di, "frames": frames}


Teacher.record_scene = record_scene


def extra_report(res, n, pairs):
    fin = {m: r[0] for m, r in res.items()}
    fins = {m: r[1] for m, r in res.items()}
    ctl = {m: r[2] for m, r in res.items()}
    days = {m: float(np.sum(fin[m] - ctl[m].start + 1)) for m in res}
    print("\n  -- 교사 추가 지표")
    for m in res:
        c = ctl[m]
        swaps = sum(sum(d.values()) for d in c.drags)
        made = np.array([np.array(f["made_tiers"])[:3] for f in fins[m]])
        chest = np.array([f["made"] for f in fins[m]])
        print(f"    {m}: 공격 무기 생산+합성 {made[:, :, :2].sum() / swaps:.3f}/스왑 (생산 {made[:, :, 0].sum() / swaps:.3f}·합성 {made[:, :, 1].sum() / swaps:.3f}) · "
              f"무기 버리기 {c.wtoss.sum() / days[m]:.2f}/게임-날 · 목적 없는 왕복 {c.rtrip.sum() / days[m]:.2f}/게임-날 · "
              f"상자 생성(일반/동/은) {'/'.join(f'{chest[:, t].sum() / days[m]:.3f}' for t in range(3))}/게임-날")
        if c.teach:
            print(f"      결정 {dict(c.stats)}")
            print(f"      교체 종류 {dict(c.kinds)} · 후보 판정 {dict(c.reject)}")
    for x, y in pairs:
        cx, cy = ctl[x], ctl[y]
        if not cx.teach:
            continue
        pl = []
        for i in range(n):
            ry = {r[0]: r for r in cy.duskrec[i]}
            for r in cx.duskrec[i]:
                if r[0] in cx.tday[i] and r[0] in ry:
                    q = ry[r[0]]
                    pw = lambda z: (z[1] * np.array([1, 3, 9, 27])).sum()
                    pl.append((pw(r) - pw(q), r[1][:, 0].sum() - q[1][:, 0].sum(), r[1][:, 1:].sum() - q[1][:, 1:].sum(), r[2] - q[2],
                               cx.night[i].get(r[0], np.nan) - cy.night[i].get(r[0], np.nan)))
        if pl:
            pr = np.array(pl)
            ok = ~np.isnan(pr[:, 4])
            print(f"    교사가 개입한 날의 밤 직전 ({x} − {y}, 같은 게임·같은 날 {len(pr)}쌍): 기본 공격 무기 {se(pr[:, 1])} · 동+ {se(pr[:, 2])} · "
                  f"전력 {se(pr[:, 0])} · 하트 {se(pr[:, 3])} · 밤 피해 {se(pr[ok, 4])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--ncand", type=int, default=6)
    p.add_argument("--z", type=float, default=2.0)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    res = {}
    for name, teach in (("P", False), ("T", True)):
        t0 = time.time()
        c = Teacher(args.games, pol, args.seed, teach, args.S, args.ncand, args.z)
        fin, fins, _ = run(pol, args.games, args.seed, rule, None, ctl=c)
        c.finish()
        res[name] = (fin, fins, c)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    labels = {"P": "P 기존 정책", "T": "T 교사"}
    print(f"\n=== {args.ckpt} · 정상 시작 {args.games}판 (시드 {args.seed}) · 대포 방향 금지 · 상자 합성 계획 켬 · 교사 S={args.S}, 후보 {args.ncand}, z={args.z} ===")
    report(res, args.games, labels, [("T", "P")])
    extra_report(res, args.games, [("T", "P")])
    if args.out:
        with open(args.out + "_decisions.json", "w") as f:
            json.dump(res["T"][2].dec, f, ensure_ascii=False, default=int)
        with open(args.out + "_games.json", "w") as f:
            json.dump({m: {"final": r[0].tolist(), "start": r[2].start.tolist(), "tday": [sorted(x) for x in r[2].tday]} for m, r in res.items()}, f)


if __name__ == "__main__":
    main()
