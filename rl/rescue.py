"""실패 장면 되돌리기: 탐색 에이전트가 30일 이후 죽은 게임을 사망한 날 아침(과 며칠 전 아침)으로 되돌려,
같은 상태를 가상 난수 여러 개로 복제한 뒤 ① 정책 혼자 ② 기존 탐색 ③ 강한 탐색으로 이어 두어 비교한다.
- 사망 당일에서 강한 탐색이 많이 살리면 "직전에 바꿀 좋은 수가 있다", 며칠 전에서만 살리면 "성장이 이미 늦었다".
- 기록된 행동을 같은 시드의 환경에 다시 두면 엔진이 결정적이라 같은 상태가 복원된다(복원 확인 포함).
또 30일 밤 직전 보드를 이후 오래 산 판과 곧 죽은 판으로 나눠 비교한다.
사용: python rl/rescue.py rl/runs/long1/agent_40M.pt [--pool 96] [--copies 8] [--back 0,3]"""
import argparse
import json
import time

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W
from search_eval import Policy, board_info, play

AGENTS = (("정책 혼자", False, 0, 0, 0), ("기존 탐색", True, 3, 2, 8), ("강한 탐색", True, 5, 6, 16))


def restore(frames, seed, points, M):
    """같은 시드 환경에 기록된 행동을 다시 두며 시점 points[q] = (게임, 프레임 번호)의 상태를
    SearchEnv 슬롯 q·M .. (q+1)·M에 가상 난수 M개로 복제해 모은다(일차·하트·스왑이 기록과 같은지 확인)"""
    n = len(frames)
    env = ts.VecEnv(n, seed=seed)
    grid, scal, mask = np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    env.reset(grid, scal, mask)
    hold = ts.SearchEnv(len(points) * M)
    at = {}
    for q, (i, t) in enumerate(points):
        at.setdefault(t, []).append((q, i))
    for t in range(max(t for _, t in points) + 1):
        for q, i in at.get(t, []):
            day, hearts, swaps, _, _ = env.state(i)
            f = frames[i][t]
            assert (day, hearts, swaps) == (f["d"], f["h"], f["s"]), f"복원 실패: 게임 {i} 행동 {t}"
            hold.store(env, np.full(M, i, np.int64), np.arange(q * M, (q + 1) * M, dtype=np.int64),
                       np.random.default_rng([seed, q, 5]).integers(1, 2 ** 62, size=M, dtype=np.uint64))
        act = np.array([frames[i][t]["a"] if t < len(frames[i]) else int(np.argmax(mask[i])) for i in range(n)], np.int64)
        env.step(act, grid, scal, mask, rew, done, days)
        env.pop_finished()
    print("복원 확인 완료 (되돌린 상태의 일차·하트·스왑이 기록과 같음)", flush=True)
    return hold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--pool", type=int, default=96)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--copies", type=int, default=8, help="되돌린 상태마다 가상 난수 복제 수")
    p.add_argument("--back", default="0,3", help="사망일로부터 며칠 전 아침으로 되돌릴지")
    p.add_argument("--min-day", type=int, default=30)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n, M = args.pool, args.copies
    backs = [int(x) for x in args.back.split(",")]

    # 1. 탐색 에이전트로 게임을 두며 행동을 기록
    t0 = time.time()
    frames = [[] for _ in range(n)]
    stats = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
    final = play(pol, n, args.seed, True, 3, 2, 8, stats, 2.0, frames=frames)
    print(f"탐색 에이전트 {n}판: day 평균 {final.mean():.2f} · 30일 도달 {np.mean(final >= 30):.0%} · 30일 통과 {np.mean(final > 30):.0%} ({time.time() - t0:.0f}s)", flush=True)
    print("  사망일 분포:", dict(sorted({int(d): int((final == d).sum()) for d in np.unique(final)}.items())), flush=True)

    # 2. 되돌릴 시점: 사망일 D ≥ min-day인 게임의 (D - b)일 첫 행동 직전
    points = []  # (게임, 프레임 번호, 사망일, 되돌린 일수)
    for i in np.nonzero(final >= args.min_day)[0]:
        D = int(final[i])
        for b in backs:
            t = next((k for k, f in enumerate(frames[i]) if f["d"] == D - b), None)
            if t is not None:
                points.append((int(i), t, D, b))
    P = len(points)
    print(f"되돌릴 시점 {P}개 (게임 {len(set(q[0] for q in points))}, 복제 {M}개씩)", flush=True)

    # 3. 같은 시드 환경에 기록된 행동을 다시 두며 되돌릴 시점의 상태를 모은다
    hold = restore(frames, args.seed, [(i, t) for i, t, D, b in points], M)

    # 4. 같은 상태에서 세 에이전트로 이어 둔다 (정책 샘플링 잡음은 에이전트끼리 같다)
    res = {}
    for name, srch, k, r, s in AGENTS:
        t0 = time.time()
        e2 = ts.VecEnv(P * M, seed=1)
        e2.load_from(hold, np.arange(P * M, dtype=np.int64), np.arange(P * M, dtype=np.int64))
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        res[name] = play(pol, P * M, args.seed + 7, srch, k, r, s, st, 2.0, env=e2).reshape(P, M)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    print("\n[되돌려 다시 두기] 원래 사망한 밤을 넘긴 비율 · 평균 최종 일차 (원래 = 사망일)")
    for b in backs:
        sel = [q for q, pt in enumerate(points) if pt[3] == b]
        if not sel:
            continue
        D = np.array([points[q][2] for q in sel])[:, None]
        print(f"  사망 {b}일 전 아침으로 ({len(sel)}개 시점, 원래 평균 사망일 {D.mean():.1f}):")
        for name, *_ in AGENTS:
            R = res[name][sel]
            saved_any = np.mean((R > D).any(1))
            print(f"    {name:6s}: 그 밤 생존 {np.mean(R > D):5.1%} · 평균 최종 {R.mean():.2f}일 · 한 번이라도 넘긴 시점 {saved_any:.0%}")

    # 5. 30일 밤 직전 보드: 이후 오래 산 판(≥35) vs 곧 죽은 판(31~33), 그리고 30일 보스에서 죽은 판
    print("\n[30일 밤 직전 보드] 하트 · 타워 점유율 · 공격 타워 기본/동/은 · 대포 수(왼/오) · 포탑 · 상자")
    groups = {"30일 보스 사망": lambda d: d == 30, "31~33일 사망": lambda d: 31 <= d <= 33, "34일 이상": lambda d: d >= 34}
    for name, cond in groups.items():
        rows = []
        for i in range(n):
            if not cond(final[i]):
                continue
            f = next((f for f in frames[i] if f["d"] == 30 and f["a"] == 224 and f["p"] == 25), None)
            if f is None:
                continue
            cells = [f["b"][r * 6:(r + 1) * 6] for r in range(7)]
            occ, tiers = board_info(cells, f["t"])
            cannons = [c for c in f["b"] if c[:1] == "c"]
            rows.append((f["h"], occ, tiers[1], tiers[2], tiers[3], sum(c.endswith("<") for c in cannons), sum(c.endswith(">") for c in cannons),
                         sum(1 for t_ in f["t"] if t_), sum(1 for c in f["b"] if c[:1] == "h")))
        if rows:
            m = np.mean(rows, 0)
            print(f"  {name} ({len(rows)}판): 하트 {m[0]:.1f} · 점유율 {m[1]:.0%} · 타워 {m[2]:.1f}/{m[3]:.1f}/{m[4]:.1f} · 대포 {m[5]:.1f}/{m[6]:.1f} · 포탑 {m[7]:.1f} · 상자 {m[8]:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"final": final.tolist(), "points": points, "res": {k: v.tolist() for k, v in res.items()}}, f)


if __name__ == "__main__":
    main()
