"""상자 경제 평가: 정책 혼자(판·스텝별 잡음 고정, search_eval의 기본 조건과 같음) 게임을 끝까지 두며 잰다.
- 성능: 생존일, 점수, 20일 보스 통과, 30일 도달
- 상자: 등급별 생성·개봉(게임당), 동상자 이상 도달 게임 비율, 합성 기회에서 합성 선택률
- 첫 동상자 생성일·첫 동상자 개봉일(그런 게임만의 평균·중앙값과 게임 비율)
- 일차 구간별 상자 개봉 스왑 수입: 그 구간에 살아 있던 게임-날 하루당 (등급별 개봉 스왑 2/12/70/380)
사용: python rl/eval_chest.py ckpt [--games 1024] [--no-open-normal] [--no-toss-day-off]"""
import argparse
import time

import numpy as np
import torch

from search_eval import Policy, play

CHEST_SWAPS = {1: 2, 2: 12, 3: 70, 4: 380}
BUCKETS = ((1, 10), (11, 20), (21, 30), (31, 99))


class ChestLog:
    """play()의 ctl: 낮(입력 대기·밤 직전)에 상자를 탭한 행동을 (게임, 일차, 등급)으로 기록한다"""

    def __init__(self):
        self.opens = []

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        if not 168 <= a < 216:
            return
        day, _, _, phase, _ = env.state(int(i))
        y, x = (a - 168) // 6, (a - 168) % 6 + 1
        if phase not in (0, 25) or y < 1:
            return
        cell = env.board(int(i))[0][y - 1][x - 1]
        if cell[:1] == "h":
            self.opens.append((int(i), day, int(cell[1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-open-normal", action="store_true", help="일반(1등급) 상자 개봉 금지")
    p.add_argument("--no-toss-day-off", action="store_true", help="휴식일 버리기 금지")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.games
    t0 = time.time()
    log, fins = ChestLog(), []
    fin = play(pol, n, args.seed, False, 0, 0, 0, None, ctl=log, no_toss=args.no_toss_day_off, no_open=args.no_open_normal, fin_rec=fins)
    fins.sort(key=lambda f: f["env"])
    made = np.array([f["made"] for f in fins])      # 게임 × 등급 1..4
    opened = np.array([f["opened"] for f in fins])
    opps, taken = sum(f["merge_opps"] for f in fins), sum(f["merge_taken"] for f in fins)
    first_made = np.array([f["first_merge_day"] + 1 if f["first_merge_day"] >= 0 else np.nan for f in fins], float)
    first_open = np.full(n, np.nan)
    for i, d, t in log.opens:
        if t == 2 and np.isnan(first_open[i]):
            first_open[i] = d
    score = np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin])
    tag = ("개봉 제한" if args.no_open_normal else "제한 없음") + (" · 휴식일 버리기 금지" if args.no_toss_day_off else "")
    print(f"{args.ckpt} ({tag}, 정책 혼자 {n}판, {time.time() - t0:.0f}s)")
    print(f"  성능: 점수 {score:.0f} · day {fin.mean():.2f} (중앙 {int(np.median(fin))}) · 20일 보스 통과 {np.mean(fin > 20):.1%} · "
          f"30일 도달 {np.mean(fin >= 30):.1%} · 30일 보스 통과 {np.mean(fin > 30):.1%}")
    print(f"  상자 생성/개봉 (게임당, 일반·동·은·금): {' · '.join(f'{m:.2f}/{o:.2f}' for m, o in zip(made.mean(0), opened.mean(0)))}")
    print(f"  동·은·금 생성 게임: {np.mean(made[:, 1] > 0):.0%} · {np.mean(made[:, 2] > 0):.0%} · {np.mean(made[:, 3] > 0):.0%} · "
          f"합성 기회 {opps / n:.2f}/게임 중 합성 선택 {taken / max(opps, 1):.0%}")
    fm, fo = first_made[~np.isnan(first_made)], first_open[~np.isnan(first_open)]
    print(f"  첫 동상자 생성일: 평균 {fm.mean() if len(fm) else np.nan:.1f} · 중앙 {np.median(fm) if len(fm) else np.nan:.0f} ({len(fm) / n:.0%}판) | "
          f"첫 동상자 개봉일: 평균 {fo.mean() if len(fo) else np.nan:.1f} · 중앙 {np.median(fo) if len(fo) else np.nan:.0f} ({len(fo) / n:.0%}판)")
    parts = []
    for lo, hi in BUCKETS:
        alive_days = sum(max(0, min(d, hi) - lo + 1) for d in fin)
        inc = [sum(CHEST_SWAPS[t] for _, d, t in log.opens if lo <= d <= hi and t == tier) for tier in (1, 2, 3, 4)]
        if alive_days:
            parts.append(f"{lo}~{hi if hi < 99 else ''}일 {sum(inc) / alive_days:.2f} (일반 {inc[0] / alive_days:.2f}·동 {inc[1] / alive_days:.2f}·은+ {(inc[2] + inc[3]) / alive_days:.2f})")
    print("  개봉 스왑 수입 (살아 있던 게임-날 하루당): " + " | ".join(parts))


if __name__ == "__main__":
    main()
