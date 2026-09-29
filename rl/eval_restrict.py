"""행동 제한 짝 비교 (추가 학습·보상 변경 없음). 기준은 force_silver_prep의 C(연쇄 포함 한 수 은상자 합성 + 준비 이동 1수).
같은 게임 시드·게임별 샘플링 잡음으로 정상 시작을 네 조건으로 둔다.
  A: 기준 그대로 / B: + 대포 방향 전환 금지(낮의 대포 탭) / C: + 모든 버리기 금지(엔진 drag_is_toss) / D: 둘 다
제한은 엔진 규칙(Env.no_cannon_flip, Env.no_toss: 마스크와 step)이라 상자 합성 계획의 후보·실행에도 똑같이 걸린다.
상자 개봉 규칙(은상자부터, 동상자 비상 예외 2,0,5)·무기 합성 우선(20,6)·휴식일 버리기 금지는 그대로.
기록: 생존·점수·20일 보스 통과, 밤 직전 공격 무기(종류·등급)·하트·밤 피해, 엔진 통계의 무기 생산·합성 횟수,
드래그별 결과(휴식일 제외, 복제 없이 전후 엔진 통계로 판정: 공격 무기 합성 / 생산 / 얼음벽 / 상자 / 성 보수 / 포탑 / 버리기 / 매치 없는 이동),
비상 동상자 개봉, 동상자 2·3개 동시 보유, 첫 은상자 도달.
사용: python rl/eval_restrict.py rl/runs/cn7/agent_10M.pt [--games 256] [--conds A,B,C,D]
     모델마다 다른 체크포인트·제한: --spec 이름=체크포인트:조건 ... --pairs 이름1-이름2,... (짝 비교: 이름1 − 이름2)"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_bronze_flow import run
from force_silver_prep import Ext, se
from ppo import parse_emergency
from search_eval import Policy, act_cat

CONDS = {"A": ("기준(C)", {}), "B": ("+대포 방향 금지", {"no_cannon_flip": True}),
         "C": ("+버리기 금지", {"no_toss": True}), "D": ("+둘 다 금지", {"no_cannon_flip": True, "no_toss": True})}
CATS = ("공격 무기 합성", "공격 무기 생산", "얼음벽", "상자 생성·합성", "성 보수", "포탑 배치", "버리기", "매치 없는 이동", "그 밖")
BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 99))
KINDS = ("t", "b", "c")  # 화살탑·발리스타·대포


def attack(cells, turrets):
    """공격 무기 수 [종류 3][등급 1..4]"""
    n = np.zeros((3, 4), int)
    for c in [x for row in cells for x in row] + list(turrets):
        if c[:1] in KINDS:
            n[KINDS.index(c[0]), min(max(int(c[1]), 1), 4) - 1] += 1
    return n


class Restrict(Ext):
    def __init__(self, n):
        super().__init__(n, "C")
        self.pre = [None] * n                      # 직전 드래그 (분류, 무기 생성 통계, 상자 생성 통계, 휴식일)
        self.drags = [Counter() for _ in range(n)]  # 드래그 결과 분류 (휴식일 제외)
        self.duskrec = [[] for _ in range(n)]      # 밤 직전 (일차, 공격 무기[3][4], 하트, 얼음벽)
        self.flips = np.zeros(n, int)
        self.lastcat = [""] * n                   # 직전 드래그 분류
        self.tosses = np.zeros(n, int)

    def force(self, env, alive, logits, noise, act):
        for i in map(int, np.nonzero(alive)[0]):
            p = self.pre[i]
            if p is None:
                continue
            self.pre[i] = None
            cat0, w0, c0, off = p
            w1 = np.array(env.made(i)[0])
            dw, dc = w1 - w0, np.array(self.cur[i]["made"]) - c0
            if cat0 == "버리기":
                cat = cat0
            elif cat0 == "돌 성에 넣기":
                cat = "성 보수"
            elif cat0 == "화살탑 포탑":
                cat = "포탑 배치"
            elif dw[:3, 1:].sum() > 0:
                cat = "공격 무기 합성"
            elif dw[:3, 0].sum() > 0:
                cat = "공격 무기 생산"
            elif dw[3].sum() > 0:
                cat = "얼음벽"
            elif dc[1:].sum() > 0:
                cat = "상자 생성·합성"
            elif cat0 == "성 쪽 드래그":
                cat = "그 밖"
            else:
                cat = "매치 없는 이동"
            self.lastcat[i] = cat
            if not off:
                self.drags[i][cat] += 1
        return super().force(env, alive, logits, noise, act)

    def after(self, env, i, a):
        i = int(i)
        c = self.cur[i]
        if c["phase"] == 0 and a < 168:
            cells = env.board(i)[0]
            cat0 = act_cat(cells, 0, a)
            self.tosses[i] += cat0 == "버리기"
            off = c["day"] % 10 == 1 and env.made(i)[2] == c["day"]
            self.pre[i] = (cat0, np.array(env.made(i)[0]), np.array(c["made"]), off)
        elif c["phase"] in (0, 25) and 168 <= a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if y >= 1 and env.board(i)[0][y - 1][x - 1][:1] == "c":
                self.flips[i] += 1
        if a == 224:
            cells, tur = env.board(i)
            self.duskrec[i].append((c["day"], attack(cells, tur), c["hearts"], sum(1 for row in cells for x in row if x[:1] == "w")))
        super().after(env, i, a)


def report(res, n, labels, pairs):
    print(f"\n{'':<26s}" + "".join(f"{labels[m]:>18s}" for m in res))
    row = lambda name, f, fmt="{:.2f}": print(f"  {name:<24s}" + "".join(f"{fmt.format(f(m)):>18s}" for m in res))
    rowtxt = lambda name, f: print(f"  {name:<24s}" + "".join(f"{f(m):>18s}" for m in res))
    fin = {m: r[0] for m, r in res.items()}
    fins = {m: r[1] for m, r in res.items()}
    ctl = {m: r[2] for m, r in res.items()}
    days = {m: float(np.sum(fin[m] - ctl[m].start + 1)) for m in res}
    row("생존일(day)", lambda m: fin[m].mean())
    row("점수", lambda m: np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin[m]]), "{:.0f}")
    row("20일 보스 통과", lambda m: np.mean(fin[m] > 20), "{:.1%}")
    row("10일 보스 통과", lambda m: np.mean(fin[m] > 10), "{:.1%}")
    for x, y in pairs:
        d = fin[x] - fin[y]
        print(f"  짝 {x} − {y}: 생존 {se(d)}일 · 점수 {np.mean([a + 1000 * min(5, (a - 1) // 10) for a in fin[x]]) - np.mean([a + 1000 * min(5, (a - 1) // 10) for a in fin[y]]):+.0f} · "
              f"20일 통과 {np.mean(fin[x] > 20) - np.mean(fin[y] > 20):+.1%} · 더 오래 {np.mean(d > 0):.0%}·같음 {np.mean(d == 0):.0%}·더 짧음 {np.mean(d < 0):.0%}")
    print("  -- 확인: 제한된 행동 (게임당)")
    row("대포 방향 전환", lambda m: ctl[m].flips.mean())
    row("버리기(휴식일 포함)", lambda m: ctl[m].tosses.mean())
    print("  -- 드래그 결과 (게임-날당 스왑, 휴식일 제외)")
    for cat in CATS:
        row(cat, lambda m, cat=cat: sum(d[cat] for d in ctl[m].drags) / days[m])
    row("합계", lambda m: sum(sum(d.values()) for d in ctl[m].drags) / days[m])
    print("  -- 엔진 통계: 무기 생성 (게임-날당)")
    made = {m: np.array([np.array(f["made_tiers"])[:, -4:] for f in fins[m]]) for m in res}  # [게임, 종류 5, 등급 1..4]
    row("공격 무기 생산(기본)", lambda m: made[m][:, :3, 0].sum() / days[m])
    row("공격 무기 합성(동+)", lambda m: made[m][:, :3, 1:].sum() / days[m])
    row("  그중 은·금", lambda m: made[m][:, :3, 2:].sum() / days[m], "{:.3f}")
    row("얼음벽 생성", lambda m: made[m][:, 3, :].sum() / days[m])
    for lo, hi in BUCKETS[:4]:
        print(f"  -- 밤 직전 ({lo}~{hi}일, 그날 밤을 맞은 게임 평균)")
        recs = {m: [(r, i) for i in range(n) for r in ctl[m].duskrec[i] if lo <= r[0] <= hi] for m in res}
        cnt = lambda m: max(1, len(recs[m]))
        row("게임-밤 수", lambda m: len(recs[m]), "{:.0f}")
        rowtxt("공격 무기 기본/동/은+", lambda m: "/".join(f"{sum(r[1][:, t].sum() for r, _ in recs[m]) / cnt(m):.1f}" for t in (0, 1))
               + f"/{sum(r[1][:, 2:].sum() for r, _ in recs[m]) / cnt(m):.1f}")
        rowtxt("화살/발리/대포", lambda m: "/".join(f"{sum(r[1][k].sum() for r, _ in recs[m]) / cnt(m):.1f}" for k in range(3)))
        row("전력(기본 환산)", lambda m: sum((r[1] * np.array([1, 3, 9, 27])).sum() for r, _ in recs[m]) / cnt(m))
        row("얼음벽", lambda m: sum(r[3] for r, _ in recs[m]) / cnt(m))
        row("밤 직전 하트", lambda m: sum(r[2] for r, _ in recs[m]) / cnt(m))
        row("밤 피해", lambda m: np.mean([ctl[m].night[i].get(r[0], np.nan) for r, i in recs[m]]) if recs[m] else np.nan)
    print("  -- 같은 게임·같은 날 짝 (두 조건 모두 그날 밤을 맞음)")
    for m, base in pairs:
        pl = []
        for i in range(n):
            a_ = {r[0]: r for r in ctl[base].duskrec[i]}
            for r in ctl[m].duskrec[i]:
                if r[0] in a_ and r[0] <= 20:
                    q = a_[r[0]]
                    pw = lambda z: (z[1] * np.array([1, 3, 9, 27])).sum()
                    pl.append((pw(r) - pw(q), r[2] - q[2], ctl[m].night[i].get(r[0], np.nan) - ctl[base].night[i].get(r[0], np.nan)))
        pr = np.array(pl)
        ok = ~np.isnan(pr[:, 2])
        print(f"    {m} − {base}: 20일까지 {len(pr)}쌍 · 전력 {se(pr[:, 0])} · 밤 직전 하트 {se(pr[:, 1])} · 밤 피해 {se(pr[ok, 2])}")
    print("  -- 상자")
    row("비상 동상자 개봉/게임", lambda m: np.mean([f["emergency_opens"] for f in fins[m]]))
    row("동상자 2개 동시 보유", lambda m: np.mean([g["held2"] is not None for g in ctl[m].g]), "{:.1%}")
    row("동상자 3개 동시 보유", lambda m: np.mean([g["held3"] is not None for g in ctl[m].g]), "{:.1%}")
    row("첫 은상자 개봉", lambda m: np.mean([f["opened"][2] > 0 for f in fins[m]]), "{:.1%}")
    rowtxt("합성 개입 / 준비 개입", lambda m: f"{ctl[m].k.sum()} / {ctl[m].kp.sum()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default="")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--conds", default="A,B,C,D")
    p.add_argument("--spec", nargs="*", default=[], help="이름=체크포인트:조건(A~D) 목록. 주면 ckpt·--conds 대신 쓴다")
    p.add_argument("--pairs", default="", help="짝 비교 '이름1-이름2,...' (기본: 첫 조건 대 나머지)")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    base = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6)}
    if args.spec:
        specs = [(sp.split("=")[0], *sp.split("=")[1].rsplit(":", 1)) for sp in args.spec]
    else:
        specs = [(m, args.ckpt, m) for m in args.conds.split(",")]
    labels = {name: f"{name} ({CONDS[cond][0]})" if args.spec else f"{name} {CONDS[cond][0]}" for name, _, cond in specs}
    pairs = [tuple(x.split("-")) for x in args.pairs.split(",") if x] or [(name, specs[0][0]) for name, _, _ in specs[1:]]
    res, pols = {}, {}
    for name, ckpt, cond in specs:
        t0 = time.time()
        pol = pols.setdefault(ckpt, Policy(ckpt, dev))
        c = Restrict(args.games)
        fin, fins, _ = run(pol, args.games, args.seed, {**base, **CONDS[cond][1]}, None, ctl=c)
        c.finish()
        res[name] = (fin, fins, c)
        print(f"  {name}: {ckpt} · {CONDS[cond][0]} · {time.time() - t0:.0f}s", flush=True)
    print(f"\n=== 정상 시작 {args.games}판 (시드 {args.seed}) · 기준 = 연쇄 포함 한 수 은상자 합성 + 준비 이동 1수 ===")
    for name, ckpt, cond in specs:
        print(f"  {name}: {ckpt} · {CONDS[cond][0]}")
    report(res, args.games, labels, pairs)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({m: {"final": r[0].tolist(), "drags": [dict(d) for d in r[2].drags]} for m, r in res.items()}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
