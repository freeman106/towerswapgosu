"""강제 합성 실험: 정책은 그대로 두고, 지정한 일차 이후 처음 오는 "교환 한 번으로 은 타워(화살탑·발리스타·대포)가
되는 합성" 기회에 한 번만 합성 교환을 강제한다. 게임을 환경마다 한 판씩, 정책 샘플링 잡음(검벨)도 판·스텝별로 고정해서
두므로 기본 조건과 개입 조건의 같은 번호 게임은 개입 직전까지 똑같다(짝 비교).
사용: python rl/force_merge.py rl/runs/cb9/agent.pt [--games 1024] [--from-days 15,20]"""
import argparse

import numpy as np
import torch

import towerswap as ts
from diag_tiers import merge_actions, parse
from ppo import A, C, H, S, W, Agent

A_END_DAY = 224
# 명세 §9 실제 발사 주기(프레임)와 피해: 화력 = 피해 × 60 / 주기 (사거리·위치는 무시)
PERIOD = {"t": (55, 23, 10, 4), "b": (196, 82, 35, 15), "c": (98, 41, 18, 8)}
DAMAGE = {"t": 3, "b": 6, "c": 6}


def board_stats(cells, turrets):
    """타워 점유율(땅 칸 중 화살탑·발리스타·대포·얼음벽 칸), 공격 타워 화력, 등급별 공격 타워 수"""
    land = towers = 0
    dps, tiers = 0.0, [0] * 5
    for row in cells:
        for s in row:
            if s == "~~":
                continue
            land += 1
            if s[:1] in ("t", "b", "c", "w"):
                towers += 1
    for s in [c for row in cells for c in row] + list(turrets):
        if s[:1] in PERIOD:
            t = min(max(int(s[1]), 1), 4)
            dps += DAMAGE[s[0]] * 60 / PERIOD[s[0]][t - 1]
            tiers[t] += 1
    return towers / land, dps, tiers


def run(agent, dev, n, seed, from_day, watch_days):
    """게임 n판을 둔다. from_day > 0이면 그날 이후 첫 은 합성 기회에 한 번 강제한다.
    watch_days의 각 일차 D에 대해, D일 이후 첫 은 합성 기회 시점(강제 여부와 관계없이)을 기록한다."""
    env = ts.VecEnv(n, seed=seed)
    grid = np.zeros((n, C, H, W), np.float32)
    scal = np.zeros((n, S), np.float32)
    mask = np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    env.reset(grid, scal, mask)
    alive = np.ones(n, bool)
    final = [None] * n
    dusk = [dict() for _ in range(n)]  # 일차 → (하트, 화력, 등급별 수)
    morning = [dict() for _ in range(n)]
    first_opp = {d: [None] * n for d in watch_days}  # 기회 시점 (일차, 하트, 점유율, 화력)
    forced = [False] * n
    step = 0
    while alive.any():
        with torch.no_grad():
            logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            logits = logits.masked_fill(~torch.from_numpy(mask).to(dev), -1e9).cpu().numpy()
        g = np.random.default_rng([seed, step]).gumbel(size=(n, A))
        act = (logits + g).argmax(1)
        for i in np.nonzero(alive)[0]:
            day, hearts, swaps, phase, _ = env.state(int(i))
            if act[i] == A_END_DAY:
                cells, tur = env.board(int(i))
                _, dps, tiers = board_stats(cells, tur)
                dusk[i][day] = (hearts, dps, tiers)
            if phase != 0 or swaps < 1:
                continue
            need = [d for d in watch_days if day >= d and first_opp[d][i] is None]
            if not need and not (from_day and day >= from_day and not forced[i]):
                continue
            cells, tur = env.board(int(i))
            silver = [a for a, (k, t) in merge_actions(parse(cells)).items() if k in "tbc" and t == 3 and mask[i, a]]
            if not silver:
                continue
            occ, dps, _ = board_stats(cells, tur)
            for d in need:
                first_opp[d][i] = (day, hearts, occ, dps)
            if from_day and day >= from_day and not forced[i]:
                act[i] = max(silver, key=lambda a: logits[i, a])
                forced[i] = True
        env.step(act, grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            i = f["env"]
            if alive[i]:
                final[i] = f["day"]
                alive[i] = False
        for i in np.nonzero((days > 0) & alive)[0]:
            day, hearts, _, _, _ = env.state(int(i))
            morning[i][day] = hearts
        step += 1
    return np.array(final), dusk, morning, first_opp, np.array(forced)


def after(dusk, morning, final, day, k=3):
    """개입일부터 k밤 동안 잃은 하트 합, 개입일 밤 직전 대비 k일 뒤 밤 직전 화력 변화와 공격 타워 수 변화(살아 있는 경우)"""
    lost, d = 0, day
    for d in range(day, day + k):
        if d not in dusk:
            break
        h = dusk[d][0]
        lost += h if d == final else h - morning.get(d + 1, h)
        if d == final:
            break
    a, b = dusk.get(day), dusk.get(day + k)
    ddps = b[1] - a[1] if a and b else None
    dcount = sum(b[2]) - sum(a[2]) if a and b else None
    return lost, ddps, dcount


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--from-days", default="15,20")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    agent.load_state_dict(ck["agent"])
    agent.eval()
    fds = [int(x) for x in args.from_days.split(",")]

    def summary(name, fin):
        r30 = fin >= 30
        return (f"{name}: day {fin.mean():.2f} · 점수 {np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin]):.0f} · "
                f"30일 도달 {r30.mean():.1%} · 도달 후 통과 {(fin > 30).sum() / max(r30.sum(), 1):.1%}")

    base, bdusk, bmorn, bopp, _ = run(agent, dev, args.games, args.seed, 0, fds)
    print(summary("기본", base))
    for fd in fds:
        fin, fdusk, fmorn, _, forced = run(agent, dev, args.games, args.seed, fd, [])
        sel = np.array([bopp[fd][i] is not None for i in range(args.games)])
        assert (sel == forced).all(), "짝 비교가 어긋났다 (개입 전 게임이 기본 조건과 다르다)"
        print(f"\n== {fd}일 이후 첫 은 합성 기회에 강제 (적격 {sel.sum()}게임, {sel.mean():.0%})")
        print("  " + summary("전체 게임", fin))
        print("  적격 게임만 짝 비교:")
        print("    " + summary("기본", base[sel]))
        print("    " + summary("강제", fin[sel]))
        opp = [bopp[fd][i] for i in np.nonzero(sel)[0]]
        print(f"    개입 시점: {np.mean([o[0] for o in opp]):.1f}일 · 하트 {np.mean([o[1] for o in opp]):.1f} · "
              f"타워 점유율 {np.mean([o[2] for o in opp]):.0%} · 화력 {np.mean([o[3] for o in opp]):.0f}")
        diff = fin[sel] - base[sel]
        print(f"    생존일 변화(강제 - 기본): 평균 {diff.mean():+.2f} · 늘어남 {np.mean(diff > 0):.0%} · 같음 {np.mean(diff == 0):.0%} · 줄어듦 {np.mean(diff < 0):.0%}")
        occ = np.array([o[2] for o in opp])
        for lo, hi in ((0, .65), (.65, .75), (.75, 1.01)):
            m = (occ >= lo) & (occ < hi)
            if m.sum():
                print(f"    점유율 {lo:.0%}~{min(hi, 1):.0%} ({m.sum()}게임): 생존일 변화 {diff[m].mean():+.2f} · "
                      f"30일 도달 {np.mean(base[sel][m] >= 30):.0%} → {np.mean(fin[sel][m] >= 30):.0%}")
        rows = {"기본": [], "강제": []}
        for i in np.nonzero(sel)[0]:
            day = bopp[fd][i][0]
            rows["기본"].append(after(bdusk[i], bmorn[i], base[i], day))
            rows["강제"].append(after(fdusk[i], fmorn[i], fin[i], day))
        for k, v in rows.items():
            dd = [x[1] for x in v if x[1] is not None]
            dc = [x[2] for x in v if x[2] is not None]
            print(f"    {k} 이후 3밤: 잃은 하트 {np.mean([x[0] for x in v]):.2f} · 3일 뒤 화력 변화 {np.mean(dd):+.1f} · "
                  f"공격 타워 수 변화 {np.mean(dc):+.2f} (3일 뒤 생존 {len(dd)}게임)")


if __name__ == "__main__":
    main()
