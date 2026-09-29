"""한 수 은상자 합성 우선 짝 비교 (추가 학습 없음). 같은 상태·같은 난수·같은 샘플링 잡음으로
기존 정책 / 우선(지금 유효한 드래그 한 번으로 은상자 이상이 새로 생기고 그 직후 그 상자를 열 수 있으면 그 드래그를 둔다)을 비교한다.
판정은 엔진 복제본(VecEnv.silver_merge_drags: 마지막 스왑을 써 밤 직전이 돼도 개봉 탭이 유효한지 포함). 스왑 기준 없음.
후보가 여럿이면 정책 확률을 후보끼리 재정규화(같은 잡음으로 argmax). 정책이 스스로 그 합성을 고르면 개입으로 세지 않는다.
개봉 규칙·비상 예외·무기 합성 우선 규칙은 그대로 둔다(무기 합성 규칙이 작동하는 순간에는 상자 합성 드래그가 마스크에 없다).
사용: python rl/force_chest_merge.py ckpt [--layouts bronze3,bronze2] [--normal]"""
import argparse
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from eval_cycle import Cycle
from eval_stage23 import LEVEL, BronzeStage
from ppo import A, C, H, S, W, parse_emergency
from search_eval import Policy, play
from starts import capture


def silver_candidates(env, i, buf):
    """동상자(또는 은상자)가 셋 이상일 때만 엔진 판정을 부른다"""
    flat = [c for row in env.board(int(i))[0] for c in row]
    if flat.count("h2") < 3 and flat.count("h3") < 3:
        return np.zeros(0, int)
    env.silver_merge_drags(int(i), buf)
    return np.nonzero(buf)[0]


class Priority:
    """choose(): 한 수 은상자 합성 후보가 있고 정책이 고른 수가 후보가 아니면 후보 중에서 고른다"""

    def init_priority(self, n):
        self.k = np.zeros(n, int)
        self.own = np.zeros(n, int)
        self.buf = np.zeros(168, bool)

    def choose(self, env, alive, logits, noise, act):
        forced = {}
        for i in np.nonzero(alive)[0]:
            if env.state(int(i))[3] != 0:
                continue
            cand = silver_candidates(env, i, self.buf)
            if len(cand) == 0:
                continue
            if act[i] in cand:
                self.own[i] += 1
                continue
            forced[int(i)] = int(cand[np.argmax(logits[i, cand] + noise[i, cand])])
            self.k[i] += 1
        return forced


class PriorityStage(BronzeStage, Priority):
    def __init__(self, n):
        BronzeStage.__init__(self, n)
        self.init_priority(n)


class PriorityCycle(Cycle, Priority):
    def __init__(self, n):
        Cycle.__init__(self, n)
        self.init_priority(n)


def run_layout(pol, args, hold, k, rule, layout, priority):
    n = args.games
    env = ts.VecEnv(n, seed=args.seed, no_toss_day_off=True, **rule)
    env.add_starts_layout(hold, np.arange(k, dtype=np.int64), LEVEL[layout], layout, seed=args.pool_seed)
    env.set_start_frac(1.0)
    env.reset(np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool))
    c, fins = (PriorityStage(n) if priority else BronzeStage(n)), []
    fin = play(pol, n, args.seed, False, 0, 0, 0, None, env=env, ctl=c, fin_rec=fins)
    fins.sort(key=lambda f: f["env"])
    return fin, fins, c


def stage_metrics(fin, fins, c, n):
    life = fin - c.start
    made = np.array([m is not None for m in c.made])
    opened = np.array([o is not None for o in c.opened])
    sameday = np.array([o is not None and o[1] == 0 for o in c.opened])
    third = np.array([f["made"][1] > 0 for f in fins])
    lost = Counter(c.lost[i] for i in range(n) if not made[i] and c.lost[i])
    held = sum(1 for i in range(n) if not made[i] and not c.lost[i])
    return {"life": life, "made": made, "opened": opened, "sameday": sameday, "third": third, "lost": lost, "held": held}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--layouts", default="bronze3,bronze2")
    p.add_argument("--normal", action="store_true", help="정상 시작에서도 비교 (시드 --normal-seed, --normal-games판)")
    p.add_argument("--normal-games", type=int, default=1024)
    p.add_argument("--normal-seed", type=int, default=777)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--open-min-tier", type=int, default=3)
    p.add_argument("--emergency", default="2,0,5")
    p.add_argument("--merge-rule", default="20,6")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    rule = {"open_min_tier": args.open_min_tier, "emergency": parse_emergency(args.emergency),
            "merge_rule": tuple(int(x) for x in args.merge_rule.split(",")) if args.merge_rule else None}
    pol = Policy(args.ckpt, dev)
    n = args.games
    if args.layouts:
        hold, k, _ = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
    for layout in [x for x in args.layouts.split(",") if x]:
        base = stage_metrics(*run_layout(pol, args, hold, k, rule, layout, False), n)
        fin, fins, c = run_layout(pol, args, hold, k, rule, layout, True)
        pri = stage_metrics(fin, fins, c, n)
        sel = np.nonzero(c.k > 0)[0]
        print(f"\n=== {args.ckpt} · {layout} · {n}판 · 개입이 일어난 판 {len(sel)} (개입 {c.k.sum()}번, 정책이 스스로 합성한 판 {np.sum(c.own > 0)}) ===")
        print(f"  {'':<24s}{'기존 정책':>12s}{'한 수 합성 우선':>16s}")
        rows = [("은상자 합성", "made"), ("합성 은상자 개봉", "opened"), ("당일 합성·개봉", "sameday")]
        if layout == "bronze2":
            rows.append(("세 번째 동상자 생성", "third"))
        for name, key in rows:
            print(f"  {name:<24s}{base[key].mean():>12.1%}{pri[key].mean():>16.1%}")
        if layout == "bronze2":
            for m, lab in ((base, "기존"), (pri, "우선")):
                t = m["third"]
                print(f"  세 번째 동상자를 만든 판 중 은상자까지 ({lab}): {np.mean(m['made'][t]) if t.any() else float('nan'):.1%} ({t.sum()}판)")
        for m, lab in ((base, "기존"), (pri, "우선")):
            print(f"  합성 전 실패 ({lab}): " + ", ".join(f"{k_} {v / n:.1%}" for k_, v in m["lost"].most_common())
                  + f", 가진 채 사망 {m['held'] / n:.1%}")
        d = pri["life"] - base["life"]
        print(f"  이후 생존: 기존 {base['life'].mean():.2f} · 우선 {pri['life'].mean():.2f} · 차이 {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(n):.2f}"
              + (f" | 개입 판만 {d[sel].mean():+.2f} ± {d[sel].std(ddof=1) / np.sqrt(len(sel)):.2f} ({len(sel)}판)" if len(sel) > 1 else ""))
    if args.normal:
        nn = args.normal_games
        res = {}
        for name, ctl in (("기존 정책", Cycle(nn)), ("한 수 합성 우선", PriorityCycle(nn))):
            fins = []
            fin = play(pol, nn, args.normal_seed, False, 0, 0, 0, None, ctl=ctl, no_toss=True, fin_rec=fins, env_kw=rule)
            res[name] = (fin, ctl)
        print(f"\n=== {args.ckpt} · 정상 시작 {nn}판 (시드 {args.normal_seed}) ===")
        for name, (fin, ctl) in res.items():
            score = np.mean([d_ + 1000 * min(5, (d_ - 1) // 10) for d_ in fin])
            first = np.mean(ctl.first_open >= 0)
            second = np.mean([sum(1 for _, _, t in o if t >= 3) >= 2 for o in ctl.opens])
            made3 = np.mean(ctl.silver_made >= 0)
            extra = f" · 개입 판 {np.sum(ctl.k > 0)}" if hasattr(ctl, "k") else ""
            print(f"  {name}: 점수 {score:.0f} · day {fin.mean():.2f} · 20일 보스 통과 {np.mean(fin > 20):.1%} · 30일 보스 통과 {np.mean(fin > 30):.1%} · "
                  f"은상자 생성 {made3:.1%} · 첫 개봉 {first:.1%} · 두 번째 개봉 {second:.1%}{extra}")
        (fb, _), (fp, cp) = res.values()
        d = fp - fb
        sel = np.nonzero(cp.k > 0)[0]
        print(f"  생존일 짝 차이: {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(nn):.2f}"
              + (f" | 개입 판만 {d[sel].mean():+.2f} ± {d[sel].std(ddof=1) / np.sqrt(len(sel)):.2f} ({len(sel)}판)" if len(sel) > 1 else ""))


if __name__ == "__main__":
    main()
