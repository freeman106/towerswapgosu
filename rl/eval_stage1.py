"""은상자 연습 1단계 평가: 학습에 쓰지 않은 상태 풀(다른 시드로 모은 정책의 낮 상태)에서 같은 상태·같은 난수로
은상자 있음(자원 하나를 은상자로 바꿈) / 없음을 짝지어 끝까지 둔다(정책 혼자). 차이가 그 정책이 은상자에서 뽑아낸 가치다.
- 이후 생존(최종 − 시작일), 은상자 개봉 여부·시점, 다음 은상자 생성·두 번째 개봉
- 시작 뒤 0~3일 밤 직전: 기본 환산 전력(Σ 3^(등급-1)), 하트, 고등급 비중(동 이상·은 이상 전력 비율), 은 이상 무기 수, 새 무기 생성(엔진 통계)
- 시작 뒤 4일(시작일 포함)의 스왑 사용처(하루당)
사용: python rl/eval_stage1.py ckpt [--games 512] [--pool 1024 --pool-seed 4321]"""
import argparse
import time
from collections import Counter

import numpy as np
import torch

import towerswap as ts
from diag_chest import drag_cat
from eval_cycle import SPEND, snapshot
from ppo import A, C, H, S, W, parse_emergency
from search_eval import Policy, play
from starts import capture, describe

SILVER_LEVEL = 6


class Stage1:
    """play()의 ctl: 시작일, 시작 뒤 0~3일 밤 직전 스냅숏, 상자 개봉, 첫 4일 드래그 사용처"""

    def __init__(self, n):
        self.start = np.zeros(n, int)
        self.snap = [{} for _ in range(n)]
        self.opens = [[] for _ in range(n)]
        self.spend = [Counter() for _ in range(n)]

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        i = int(i)
        day, hearts, _, phase, _ = env.state(i)
        if self.start[i] == 0:
            self.start[i] = day
        if phase not in (0, 25):
            return
        k = day - self.start[i]
        cells, tur = env.board(i)
        if a < 168:
            if phase == 0 and k <= 3 and not (day % 10 == 1 and env.made(i)[2] == day):
                self.spend[i][drag_cat(cells, a)] += 1
        elif a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if y >= 1 and cells[y - 1][x - 1][:1] == "h":
                self.opens[i].append((day, int(cells[y - 1][x - 1][1])))
        elif a == 224 and k <= 3:
            self.snap[i][k] = snapshot(env, i, cells, tur, hearts)


def run(pol, n, seed, hold, k, place, rule, pool_seed):
    env = ts.VecEnv(n, seed=seed, no_toss_day_off=True, **rule)
    env.add_starts(hold, np.arange(k, dtype=np.int64), SILVER_LEVEL, place_tier=place, seed=pool_seed)
    env.set_start_frac(1.0)
    env.reset(np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool))
    c, fins = Stage1(n), []
    fin = play(pol, n, seed, False, 0, 0, 0, None, env=env, ctl=c, fin_rec=fins)
    fins.sort(key=lambda f: f["env"])
    return fin, fins, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024, help="평가용 상태 수 (학습 풀과 다른 시드로 모은다)")
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--pool-open-min-tier", type=int, default=2)
    p.add_argument("--open-min-tier", type=int, default=3)
    p.add_argument("--emergency", default="2,0,5")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()
    hold, k, info = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool,
                            env_kw={"open_min_tier": args.pool_open_min_tier})
    pol = Policy(args.ckpt, dev)
    rule = {"open_min_tier": args.open_min_tier, "emergency": parse_emergency(args.emergency)}
    n = args.games
    (fa, ra, ca), (fb, rb, cb) = (run(pol, n, args.seed, hold, k, pl, rule, args.pool_seed) for pl in (3, 0))
    assert (ca.start == cb.start).all(), "짝이 어긋남: 있음/없음 조건의 시작 상태가 다르다"
    st = ca.start
    print(f"=== {args.ckpt} · 규칙 {args.open_min_tier}등급부터, 비상 {args.emergency or '없음'} · 평가 풀 {describe(info)} · {n}판 ({time.time() - t0:.0f}s) ===")
    la, lb = fa - st, fb - st
    d = la - lb
    print(f"  이후 생존(최종 − 시작일): 은상자 있음 {la.mean():.2f} · 없음 {lb.mean():.2f} · 차이 {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(n):.2f}"
          f" · 있음이 더 오래 {np.mean(d > 0):.0%} / 같음 {np.mean(d == 0):.0%} / 더 짧게 {np.mean(d < 0):.0%}")
    for bd in (10, 20, 30):
        sel = st <= bd
        if sel.sum() >= 20:
            print(f"    {bd}일 보스 통과 (그 전에 시작한 {sel.sum()}판): 있음 {np.mean(fa[sel] > bd):.1%} · 없음 {np.mean(fb[sel] > bd):.1%}")
    silver_open = np.array([[d_ for d_, t in o if t >= 3] for o in ca.opens], dtype=object)
    first = np.array([o[0] - s if o else -1 for o, s in zip(silver_open, st)])
    opened = first >= 0
    print(f"  배치한 은상자 개봉: {opened.mean():.1%} (시작 뒤 평균 {first[opened].mean() if opened.any() else np.nan:.1f}일, "
          f"당일 {np.mean(first == 0):.0%}) · 죽을 때까지 안 연 판 {np.mean(~opened):.0%}")
    made3a = np.array([f["made"][2] + f["made"][3] for f in ra])
    made3b = np.array([f["made"][2] + f["made"][3] for f in rb])
    second = np.array([len(o) >= 2 for o in silver_open])
    print(f"  다음 은상자: 새로 생성 있음 {np.mean(made3a > 0):.1%} · 없음 {np.mean(made3b > 0):.1%} | 두 번째 은상자 개봉 {second.mean():.1%}")
    ea, eb = np.array([f["emergency_opens"] for f in ra]), np.array([f["emergency_opens"] for f in rb])
    print(f"  비상 동상자 개봉/게임: 있음 {ea.mean():.2f} · 없음 {eb.mean():.2f}")
    print(f"  시작 뒤 밤 직전 (있음 / 없음, 둘 다 살아 있는 판만 짝지어 비교):")
    print(f"    {'날':<6s}{'판':>5s}{'전력':>14s}{'하트':>14s}{'동+ 전력 비중':>16s}{'은+ 전력 비중':>16s}{'은+ 무기':>12s}{'새 무기 기본/동/은(시작부터)':>34s}")
    for kk in range(4):
        pair = [(ca.snap[i][kk], cb.snap[i][kk], ca.snap[i].get(0), cb.snap[i].get(0)) for i in range(n) if kk in ca.snap[i] and kk in cb.snap[i]]
        if not pair:
            continue
        def m(f, j):
            return np.mean([f(x[j]) for x in pair])
        hi2 = lambda s: sum(s["atk"][t] * 3 ** t for t in range(1, 4)) / max(s["strength"], 1)
        hi3 = lambda s: sum(s["atk"][t] * 3 ** t for t in range(2, 4)) / max(s["strength"], 1)
        silver_w = lambda s: sum(s["atk"][2:])
        made = lambda j: "/".join(f"{np.mean([x[j]['made_w'][t] for x in pair]):.1f}" for t in range(3))
        print(f"    +{kk}일{len(pair):>7d}{m(lambda s: s['strength'], 0):>7.1f} / {m(lambda s: s['strength'], 1):<5.1f}"
              f"{m(lambda s: s['hearts'], 0):>7.1f} / {m(lambda s: s['hearts'], 1):<5.1f}"
              f"{m(hi2, 0):>8.0%} / {m(hi2, 1):<5.0%}{m(hi3, 0):>8.0%} / {m(hi3, 1):<5.0%}"
              f"{m(silver_w, 0):>5.1f} / {m(silver_w, 1):<4.1f}{made(0):>17s} / {made(1)}")
    for name, cc, ff in (("있음", ca, fa), ("없음", cb, fb)):
        days = sum(min(4, ff[i] - st[i] + 1) for i in range(n))
        sp = Counter()
        for x in cc.spend:
            sp.update(x)
        print(f"  첫 4일 스왑 사용처/일 ({name}): " + ", ".join(f"{c_} {sp[c_] / days:.2f}" for c_ in SPEND if sp[c_]))


if __name__ == "__main__":
    main()
