"""은상자 합성 개입 확장 짝 비교 (추가 학습 없음). 같은 상태·같은 난수·같은 샘플링 잡음으로 세 조건을 둔다.
A: 기존 한 수 은상자 합성 우선(force_chest_merge.Priority: 동상자·은상자가 3개 이상일 때 silver_merge_drags 후보)
B: 누락 보완. 동상자 수로 제한하지 않고, 지금 마스크의 드래그 중 은상자 이상이 새로 생기고 직후 열 수 있는 수
   (동상자 2개 + 세 번째 생성 연쇄, 모루 포함). 판정 복제본은 실제 난수 대신 고정 가상 시드를 쓰고 모든 시드에서 성립해야 한다
   (VecEnv.silver_safe_drags) → 실제 미래 난수를 쓰지 않고, 새 타일의 무작위 결과에 기대는 수는 빠진다
C: B + 한 수 완성이 없고 남은 스왑이 2개 이상이면 준비 이동 1수: 상자·모루를 움직이는 드래그 중, 모든 가상 시드에서 그 뒤
   같은 드래그가 은상자를 만들고 직후 열 수 있는 수(VecEnv.prep_silver_drags). 첫 수 뒤 실제 상태에서 B 판정으로 다시 확인해
   유효하면 합성한다(아니면 개입하지 않음)
후보가 여럿이면 정책 확률을 후보끼리 재정규화(같은 잡음으로 argmax). 정책이 스스로 후보를 고르면 개입으로 세지 않는다.
개봉 규칙·비상 예외·무기 합성 우선 규칙(20,6)·휴식일 버리기 금지는 그대로 둔다.
사용: python rl/force_silver_prep.py rl/runs/cn7/agent_10M.pt [--conds A,B,C] [--normal] [--skip-stage3]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_bronze_flow import Flow, run
from force_chest_merge import Priority
from ppo import parse_emergency
from search_eval import Policy
from starts import capture

SEEDS = 4  # 가상 판정 시드 수


class Ext(Flow):
    """Flow(동상자 흐름 기록) + 조건별 개입 + 밤 피해·행동열 기록"""

    def __init__(self, n, mode, audit=False):
        super().__init__(n)
        self.mode, self.audit = mode, audit
        self.kp = np.zeros(n, int)         # 준비 이동 개입 수
        self.own_prep = np.zeros(n, int)   # 정책이 스스로 준비 이동 후보를 고른 수
        self.log = [[] for _ in range(n)]  # 개입 (스텝, 일차, 종류)
        self.acts = [[] for _ in range(n)]  # 둔 행동 (행동, 일차)
        self.plan = [None] * n              # C: 준비 이동을 둔 스텝 (다음 입력 대기에서 재확인)
        self.recheck = Counter()
        self.dusk = [None] * n              # 밤 시작 직전 (일차, 하트)
        self.night = [{} for _ in range(n)]  # 일차 → 그 밤의 하트 손실
        self.diff_ab = Counter()            # B 판정과 A 판정(실제 난수 복제본) 비교: 동상자·은상자 3개 이상 상태

    def note(self, env, i, kind):
        c = self.cur[i]
        off = c["day"] % 10 == 1 and env.made(i)[2] == c["day"]
        self.log[i].append((int(self.step[i]), c["day"], kind, c["b"], bool(off)))

    def force(self, env, alive, logits, noise, act):
        for i in map(int, np.nonzero(alive)[0]):
            c, d = self.cur[i], self.dusk[i]
            if d is not None and c["day"] != d[0]:
                self.night[i][d[0]] = d[1] - c["hearts"]
                self.dusk[i] = None
        if self.mode == "A":
            forced = Priority.choose(self, env, alive, logits, noise, act)
            for i in forced:
                self.note(env, i, "합성")
            return forced
        idle = np.array([i for i in map(int, np.nonzero(alive)[0]) if self.cur[i]["phase"] == 0], np.int64)
        forced, prep = {}, []
        if not len(idle):
            return forced
        out = np.zeros((len(idle), 168), bool)
        env.silver_safe_drags(idle, out, SEEDS)
        buf = np.zeros(168, bool)
        for j, i in enumerate(map(int, idle)):
            cand = np.nonzero(out[j])[0]
            if self.audit:
                cells = self.cur[i]["cells"]
                flat = [x for row in cells for x in row] if cells else []
                if flat.count("h2") >= 3 or flat.count("h3") >= 3:
                    env.silver_merge_drags(i, buf)
                    a_set = set(np.nonzero(buf)[0])
                    self.diff_ab["같음" if a_set == set(cand) else "B만 있음" if not a_set else "A만 있음" if not len(cand) else "후보 다름"] += 1
            if self.plan[i] is not None:
                self.recheck["합성 가능" if len(cand) else "경로 깨짐"] += 1
                self.plan[i] = None
            if len(cand):
                if act[i] in cand:
                    self.own[i] += 1
                    continue
                forced[i] = int(cand[np.argmax(logits[i, cand] + noise[i, cand])])
                self.k[i] += 1
                self.note(env, i, "합성")
            elif self.mode == "C" and self.cur[i]["swaps"] >= 2:
                prep.append(i)
        if prep:
            out2 = np.zeros((len(prep), 168), bool)
            env.prep_silver_drags(np.array(prep, np.int64), out2, SEEDS)
            for j, i in enumerate(prep):
                cand = np.nonzero(out2[j])[0]
                if not len(cand):
                    continue
                self.plan[i] = int(self.step[i])
                if act[i] in cand:
                    self.own_prep[i] += 1
                    continue
                forced[i] = int(cand[np.argmax(logits[i, cand] + noise[i, cand])])
                self.kp[i] += 1
                self.note(env, i, "준비")
        return forced

    def after(self, env, i, a):
        i = int(i)
        c = self.cur[i]
        self.acts[i].append((a, c["day"]))
        if a == 224:
            self.dusk[i] = (c["day"], c["hearts"])
        super().after(env, i, a)

    def finish(self):
        for i, d in enumerate(self.dusk):
            if d is not None:  # 그 밤에 게임이 끝남
                self.night[i][d[0]] = d[1]


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.2f} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f}" if len(x) > 1 else (f"{x.mean():+.2f}" if len(x) else "-")


def summarize(name, fin, c, n):
    """조건 하나의 지표"""
    G = c.g
    life = fin - c.start
    iv = (c.k > 0) | (c.kp > 0)
    made = np.array([g["silver"] is not None for g in G])
    opened = np.array([g["opened"] is not None for g in G])
    typ = Counter(g["silver"]["type"] for g in G if g["silver"])
    od = [g["opened"]["rel_day"] for g in G if g["opened"]]
    same = [g["opened"]["day"] == g["silver"]["day"] for g in G if g["opened"]]
    lost = np.array([len(g["losses"]) > 0 for g in G])
    cause = Counter(l["cause"] for g in G for l in g["losses"])
    open_night = [c.night[i].get(G[i]["opened"]["day"]) for i in range(n) if G[i]["opened"]]
    open_night = [x for x in open_night if x is not None]
    logs = [e for L in c.log for e in L]
    swaps = sum(1 for e in logs if not e[4])
    by_b = Counter("동상자 3개 이상" if e[3] >= 3 else f"동상자 {e[3]}개(연쇄·모루)" for e in logs if e[2] == "합성")
    print(f"\n  [{name}] 개입 게임 {iv.sum()} · 합성 개입 {c.k.sum()}번(" + ", ".join(f"{k_} {v}" for k_, v in by_b.most_common())
          + f") · 준비 이동 개입 {c.kp.sum()}번 · 개입에 쓴 스왑 {swaps}(휴식일 제외) · 정책이 스스로 후보를 고른 판 합성 {np.sum(c.own > 0)}·준비 {np.sum(c.own_prep > 0)}")
    print(f"    은상자 합성 {made.mean():.1%} (직접 {typ['직접']}·연쇄 {typ['연쇄']}) · 합성 은상자 개봉 {opened.mean():.1%} · "
          f"개봉 시점 시작 뒤 평균 {np.mean(od) if od else float('nan'):.2f}일 (중앙 {np.median(od) if od else float('nan'):.0f}) · 합성 당일 개봉 {np.mean(same) if same else float('nan'):.0%}")
    print(f"    합성 전 동상자 손실 {lost.mean():.1%} (판당 {np.mean([len(g['losses']) for g in G]):.2f}건: "
          + ", ".join(f"{k_} {v}" for k_, v in cause.most_common()) + ")")
    print(f"    개봉일 밤 피해 {np.mean(open_night) if open_night else float('nan'):.2f} ({len(open_night)}판) · 이후 생존 {life.mean():.2f}일 · "
          f"20일 보스 통과 {np.mean(fin > 20):.1%} · 30일 보스 통과 {np.mean(fin > 30):.1%}")
    if c.recheck:
        print(f"    준비 이동 뒤 실제 상태 재확인: " + ", ".join(f"{k_} {v}" for k_, v in c.recheck.most_common()))
    if c.diff_ab:
        print(f"    (점검) 동상자·은상자 3개 이상 입력 대기 상태에서 B 판정 대 A 판정: " + ", ".join(f"{k_} {v}" for k_, v in c.diff_ab.most_common()))
    return {"life": life, "made": made, "opened": opened, "lost": lost}


def paired(label, x, y, fx, fy, n):
    """x − y 짝 비교: 전체 판과, 처음 행동이 갈린 판(갈린 날의 밤 피해 포함)"""
    div = []
    for i in range(n):
        ax, ay = x.acts[i], y.acts[i]
        k = next((t for t, (u, v) in enumerate(zip(ax, ay)) if u[0] != v[0]), None)
        if k is None and len(ax) != len(ay):
            k = min(len(ax), len(ay))
        if k is not None:
            div.append((i, ax[k][1] if k < len(ax) else ay[k][1]))
    dl = (fx - x.start) - (fy - y.start)
    sel = np.array([i for i, _ in div], int)
    mx, my = (np.array([g["silver"] is not None for g in c_.g]) for c_ in (x, y))
    ox, oy = (np.array([g["opened"] is not None for g in c_.g]) for c_ in (x, y))
    lx, ly = (np.array([len(g["losses"]) > 0 for g in c_.g]) for c_ in (x, y))
    print(f"\n  [{label}] 행동이 갈린 판 {len(sel)} / {n}")
    print(f"    전체: 이후 생존 {se(dl)} · 합성 {mx.mean() - my.mean():+.1%} · 개봉 {ox.mean() - oy.mean():+.1%} · 합성 전 손실 {lx.mean() - ly.mean():+.1%}")
    if len(sel):
        nx = [x.night[i].get(d) for i, d in div]
        ny = [y.night[i].get(d) for i, d in div]
        both = [(a, b) for a, b in zip(nx, ny) if a is not None and b is not None]
        print(f"    갈린 판만: 이후 생존 {se(dl[sel])} · 합성 {mx[sel].mean():.1%} vs {my[sel].mean():.1%} · 개봉 {ox[sel].mean():.1%} vs {oy[sel].mean():.1%} · "
              f"합성 전 손실 {lx[sel].mean():.1%} vs {ly[sel].mean():.1%}")
        if both:
            d = [a - b for a, b in both]
            print(f"    갈린 날 밤 피해 {np.mean([a for a, _ in both]):.2f} vs {np.mean([b for _, b in both]):.2f} (차이 {se(d)}, 두 조건 모두 그날 밤을 맞은 {len(both)}판)")
        better, worse = int(np.sum(dl[sel] > 0)), int(np.sum(dl[sel] < 0))
        print(f"    갈린 판 중 더 오래 삶 {better} · 같음 {len(sel) - better - worse} · 더 짧음 {worse}")
    return sel


def run_all(pol, conds, n, seed, rule, starts, audit):
    res = {}
    for m in conds:
        t0 = time.time()
        c = Ext(n, m, audit=audit and m == "B")
        fin, fins, _ = run(pol, n, seed, rule, starts, ctl=c)
        c.finish()
        res[m] = (fin, fins, c)
        print(f"  조건 {m}: {time.time() - t0:.0f}s", flush=True)
    return res


def report(title, res, n, normal):
    print(f"\n=== {title} ===")
    for m, (fin, fins, c) in res.items():
        summarize(m, fin, c, n)
    if normal:
        print("\n  정상 시작 첫 은상자:")
        for m, (fin, fins, c) in res.items():
            opened = np.mean([f["opened"][2] > 0 for f in fins])
            made = np.mean([f["made"][2] > 0 for f in fins])
            score = np.mean([d_ + 1000 * min(5, (d_ - 1) // 10) for d_ in fin])
            print(f"    {m}: 은상자 생성 {made:.1%} · 첫 은상자 개봉 {opened:.1%} · 점수 {score:.0f} · day {fin.mean():.2f}")
    if "A" in res and "B" in res:
        paired("B − A: 연쇄 기회 누락 보완", res["B"][2], res["A"][2], res["B"][0], res["A"][0], n)
    if "B" in res and "C" in res:
        paired("C − B: 준비 이동 1수", res["C"][2], res["B"][2], res["C"][0], res["B"][0], n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--conds", default="A,B,C")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--normal", action="store_true", help="정상 시작에서도 비교")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--normal-games", type=int, default=1024)
    p.add_argument("--normal-seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--audit", action="store_true", help="B 조건에서 A 판정과 후보 비교")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6)}
    pol = Policy(args.ckpt, dev)
    conds = [c for c in args.conds.split(",") if c]
    out = {}
    if not args.skip_stage3:
        hold, k, _ = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
        res = run_all(pol, conds, args.games, args.seed, rule, (hold, k, args.pool_seed), args.audit)
        report(f"{args.ckpt} · 3단계(동상자 2개) {args.games}판", res, args.games, False)
        out["stage3"] = res
    if args.normal:
        res = run_all(pol, conds, args.normal_games, args.normal_seed, rule, None, args.audit)
        report(f"{args.ckpt} · 정상 시작 {args.normal_games}판 (시드 {args.normal_seed})", res, args.normal_games, True)
        out["normal"] = res
    if args.out:
        dump = {}
        for name, res in out.items():
            dump[name] = {m: [{"final": int(fin[i]), "start": int(c.start[i]), "k": int(c.k[i]), "kp": int(c.kp[i]), "log": c.log[i],
                               "silver": c.g[i]["silver"] and {k_: v for k_, v in c.g[i]["silver"].items() if k_ != "streak"},
                               "opened": c.g[i]["opened"], "losses": [{k_: v for k_, v in l.items() if k_ != "board"} for l in c.g[i]["losses"]],
                               "night": c.night[i]} for i in range(len(fin))] for m, (fin, fins, c) in res.items()}
        with open(args.out, "w") as f:
            json.dump(dump, f, ensure_ascii=False, default=int)


if __name__ == "__main__":
    main()
