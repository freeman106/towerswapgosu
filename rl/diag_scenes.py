"""장면 진단 (행동 기록 기준):
A. 버리기: 버린 뒤 타워·하트로 이어지는가, 같은 칸만 계속 버리는가, 그때 다른 매치가 가능했는가
B. 아래쪽 1·2등급: 행별로 타워 1개가 하룻밤에 주는 피해, 보드가 찼을 때 합성·새 타워 경로가 있는가
C. 성 보수: 그때 하트가 급했는가, 교환 한 번으로 화살탑을 만들 수 있었는가
D. 화살탑: 어디에 생기는가, 포탑으로 올릴 기회가 있었고 썼는가
새로 게임을 두어 통계를 내고(--games), 리플레이 기록(--replay)이 있으면 그 판들의 대표 장면(행동 번호)도 뽑는다.
사용: python rl/diag_scenes.py rl/runs/long1/agent_40M.pt [--games 128] [--replay pairs.json]"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from search_eval import Policy, play

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
RES_TO = {"l": "b", "s": "t", "i": "c", "g": "h", "d": "w"}  # 자원 3개 → 방어물 1등급
TOWERS = "tbcw"


def parse(f):
    b = {}
    for k, s in enumerate(f["b"]):
        if s not in ("..", "~~"):
            b[(k % 6 + 1, k // 6 + 1)] = (s[0], int(s[1]) if s[1].isdigit() else 0)
    return b


def line3(b, x, y):
    t = b.get((x, y))
    if t is None or not (t[0] in RES_TO and t[1] == 0 or t[0] in TOWERS + "h" and 1 <= t[1] < 4):
        return None
    for dx, dy in ((1, 0), (0, 1)):
        n = 1
        for sg in (1, -1):
            px, py = x + sg * dx, y + sg * dy
            while b.get((px, py)) == t:
                n += 1
                px, py = px + sg * dx, py + sg * dy
        if n >= 3:
            return t
    return None


def one_swap_matches(b):
    """교환 한 번으로 생기는 매치: {행동: (결과 종류, 결과 등급)} (이웃 교환만, 근사)"""
    out = {}
    for (x, y), ta in b.items():
        for d, (dx, dy) in enumerate(DIRS):
            q = (x + dx, y + dy)
            tb = b.get(q)
            if tb is None or tb == ta or "C" in (ta[0], tb[0]):
                continue
            nb = dict(b)
            nb[(x, y)], nb[q] = tb, ta
            for c in ((x, y), q):
                r = line3(nb, *c)
                if r:
                    out[((y - 1) * 6 + (x - 1)) * 4 + d] = (RES_TO[r[0]], 1) if r[1] == 0 else (r[0], r[1] + 1)
    return out


def towers(b):
    return sum(1 for k, t in b.values() if k in TOWERS)


def tier_sum(b):
    return sum(3 ** (t - 1) for k, t in b.values() if k in TOWERS and t >= 1)


def drag_kind(f, b):
    a = f["a"]
    if f["p"] != 0 or a >= 168:
        return None, None
    c = a // 4
    x, y = c % 6 + 1, c // 6 + 1
    dx, dy = DIRS[a % 4]
    tx, ty = x + dx, y + dy
    src = b.get((x, y))
    if ty == 0:
        return ("성 보수" if src and src[0] == "s" else "포탑" if src and src[0] == "t" else "성 쪽"), (x, y)
    if not (1 <= tx <= 6 and 1 <= ty <= 7) or f["b"][(ty - 1) * 6 + (tx - 1)] == "~~":
        return "버리기", (x, y)
    if (tx, ty) not in b:
        return "이동", (x, y)
    return "교환", (x, y)


def analyze(games, label):
    """games: [(frames, final_day)]"""
    A = Counter()
    A_after = Counter()
    toss_at = []  # (게임, 행동 번호, 일차, 같은 날 앞선 버리기와 같은 칸?)
    streaks = []
    swap_tower = [0, 0]
    castle = []
    arrows_new = Counter()
    turret_opp = [0, 0]
    occ_rows = []
    for gi, (F, fin) in enumerate(games):
        tossed_today, day, run, run_start = set(), None, 0, 0
        for k, f in enumerate(F):
            if f["d"] != day:
                day, tossed_today = f["d"], set()
            b = parse(f)
            kind, pos = drag_kind(f, b)
            nxt = parse(F[k + 1]) if k + 1 < len(F) else None
            # 화살탑 포탑 기회: (1,1)·(6,1)의 화살탑, 슬롯이 비었거나 더 낮은 등급
            if f["p"] == 0 and f["s"] > 0:
                for x, slot in ((1, 0), (6, 5)):
                    t = b.get((x, 1))
                    cur = f["t"][slot]
                    if t and t[0] == "t" and (not cur or int(cur[1]) < t[1]):
                        turret_opp[0] += 1
                        turret_opp[1] += int(kind == "포탑")
                        break
            if nxt is not None and f["d"] == F[k + 1]["d"]:
                before = Counter((xy, v) for xy, v in b.items() if v[0] == "t")
                for xy, v in nxt.items():
                    if v[0] == "t" and v[1] == 1 and (xy, v) not in before:
                        arrows_new[xy[1]] += 1
            if kind == "버리기":
                run += 1
                if run == 1:
                    run_start = k
                A["버리기"] += 1
                rep = pos in tossed_today
                tossed_today.add(pos)
                A["같은 날 같은 칸"] += rep
                avail = one_swap_matches(b)
                A["그때 매치 가능"] += bool(avail)
                A["그때 타워 합성 가능"] += any(t >= 2 for _, t in avail.values())
                src = b[pos]
                A["버린 것: " + ("자원" if src[1] == 0 else "방어물" if src[0] in TOWERS else src[0])] += 1
                if nxt is not None:
                    A_after["바로 타워 생김"] += towers(nxt) > towers(b) or tier_sum(nxt) > tier_sum(b)
                    # 이후 3수 안에 타워 생성 또는 하트 증가 (같은 날)
                    later = [F[j] for j in range(k + 1, min(k + 4, len(F))) if F[j]["d"] == f["d"]]
                    if later:
                        lb = parse(later[-1])
                        A_after["3수 안 타워 증가"] += tier_sum(lb) > tier_sum(b)
                        A_after["3수 안 하트 증가"] += later[-1]["h"] > f["h"]
                toss_at.append((gi, k, f["d"], rep))
            else:
                if run >= 4:
                    streaks.append((gi, run_start, run, F[run_start]["d"]))
                run = 0
            if kind == "교환" and nxt is not None:
                swap_tower[0] += 1
                swap_tower[1] += tier_sum(nxt) > tier_sum(b)
            if kind == "성 보수":
                before_m = one_swap_matches(b)
                arrow_before = any(v == ("t", 1) for v in before_m.values())
                arrow_after = nxt is not None and any(v == ("t", 1) for v in one_swap_matches(nxt).values())
                boss_in = (10 - f["d"] % 10) % 10
                castle.append((f["h"], arrow_before, arrow_before and not arrow_after, boss_in, sum(1 for v in b.values() if v == ("s", 0))))
            if f["p"] == 25 and f["a"] == 224:
                land = sum(1 for s in f["b"] if s != "~~")
                occ = sum(1 for v in b.values() if v[0] in TOWERS) / land
                m = one_swap_matches(b) if False else None
                rows = [[b.get((x, y), ("", 0)) for x in range(1, 7)] for y in range(1, 7)]
                occ_rows.append((f["d"], occ, rows))
    n = A["버리기"]
    print(f"\n===== {label} =====")
    print(f"[A. 버리기] {n}번 · 같은 날 이미 버린 칸을 또 버림 {A['같은 날 같은 칸'] / n:.0%} · "
          f"그때 교환 한 번 매치가 가능했음 {A['그때 매치 가능'] / n:.0%} (타워 합성 가능 {A['그때 타워 합성 가능'] / n:.0%})")
    print(f"  버린 것: 자원 {A['버린 것: 자원'] / n:.0%}, 방어물 {A['버린 것: 방어물'] / n:.0%}")
    print(f"  결과: 바로 타워 생김(연쇄) {A_after['바로 타워 생김'] / n:.0%} · 3수 안 타워 증가 {A_after['3수 안 타워 증가'] / n:.0%} · "
          f"3수 안 하트 증가 {A_after['3수 안 하트 증가'] / n:.0%}  (비교: 교환 한 번에 타워 생김 {swap_tower[1] / max(swap_tower[0], 1):.0%})")
    print(f"  4번 이상 연속 버리기 {len(streaks)}구간 (게임당 {len(streaks) / len(games):.1f}), 평균 길이 {np.mean([s[2] for s in streaks]) if streaks else 0:.1f}")
    c = np.array(castle, dtype=float)
    if len(c):
        hb = np.histogram(c[:, 0], bins=[0, 6, 11, 21, 31, 99])[0] / len(c)
        print(f"[C. 성 보수] {len(c)}번 · 그때 하트 1~5 {hb[0]:.0%}, 6~10 {hb[1]:.0%}, 11~20 {hb[2]:.0%}, 21~30 {hb[3]:.0%}, 31+ {hb[4]:.0%} · "
              f"보스까지 평균 {c[:, 3].mean():.1f}일")
        print(f"  그때 교환 한 번으로 화살탑을 만들 수 있었음 {c[:, 1].mean():.0%} · 이 성 보수로 그 기회가 사라짐 {c[:, 2].mean():.0%} · 보드의 돌 평균 {c[:, 4].mean():.1f}개")
    tot_new = sum(arrows_new.values())
    print(f"[D. 화살탑] 새로 생긴 기본 화살탑 {tot_new}개 (게임당 {tot_new / len(games):.1f}) · 행 분포 " +
          ", ".join(f"{r}행 {arrows_new[r] / max(tot_new, 1):.0%}" for r in range(1, 7)))
    print(f"  포탑으로 올릴 기회(1행 양 끝 화살탑, 슬롯 비었거나 낮은 등급) {turret_opp[0]}번 결정 · 실제로 올림 {turret_opp[1]}번")
    return occ_rows, streaks, toss_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--seed", type=int, default=555)
    p.add_argument("--replay", default="")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.games
    frames = [[] for _ in range(n)]
    fins = []
    import towerswap as ts
    orig = ts.VecEnv.pop_finished
    final = play(pol, n, args.seed, False, 0, 0, 0, None, frames=frames)
    occ_rows, _, _ = analyze(list(zip(frames, final)), f"{args.ckpt} 정책 혼자 {n}판 (새로 둠, day {final.mean():.2f})")

    # B. 행별 기여: 같은 설정으로 한 번 더 두어 엔진 통계를 모은다
    env = ts.VecEnv(n, seed=args.seed + 1)
    from ppo import A, C, H, S, W
    grid, scal, mask = np.zeros((n, C, H, W), np.float32), np.zeros((n, S), np.float32), np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    env.reset(grid, scal, mask)
    alive = np.ones(n, bool)
    step = 0
    while alive.any():
        logits, _ = pol(grid, scal, mask)
        act = (logits + np.random.default_rng([args.seed, step]).gumbel(size=(n, A))).argmax(1)
        env.step(act, grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            if alive[f["env"]]:
                alive[f["env"]] = False
                fins.append(f)
        step += 1
    dmg = np.sum([f["dmg_row"] for f in fins], 0)
    tn = np.sum([f["tn_row"] for f in fins], 0)
    print("\n[B. 행별 기여] 타워 1개가 하룻밤에 주는 피해 (타워-밤 수) · 1행이 성 쪽, 6행이 물 쪽(드래곤이 들어오는 쪽)")
    for k, name in enumerate(("화살탑", "발리스타", "대포")):
        print(f"  {name:5s}: " + " | ".join(f"{r}행 {dmg[k, r] / max(tn[k, r], 1):5.1f} ({tn[k, r]})" for r in range(1, 7)))
    # 밤 직전 보드의 행별 구성 (20일 이후)
    late = [r for r in occ_rows if r[0] >= 20]
    if late:
        print("  20일 이후 밤 직전 보드, 행별 평균: 기본 타워 / 동 이상 타워 / 자원·기타")
        for y in range(6):
            t1 = np.mean([sum(1 for k_, t in row[y] if k_ in TOWERS and t == 1) for _, _, row in late])
            t2 = np.mean([sum(1 for k_, t in row[y] if k_ in TOWERS and t >= 2) for _, _, row in late])
            print(f"    {y + 1}행: {t1:.1f} / {t2:.1f} / {6 - t1 - t2:.1f}")
        hi = [r for r in late if r[1] >= 0.75]
        print(f"  20일 이후 점유율 75% 이상인 밤 직전: {len(hi)}회")

    if args.replay:
        R = json.load(open(args.replay))
        for cond in ("base", "search"):
            games = [(g[cond]["frames"], g[cond]["day"]) for g in R["games"]]
            _, streaks, _ = analyze(games, f"리플레이 5판 · {'기본 정책' if cond == 'base' else '탐색'}")
            print("  리플레이에서 볼 연속 버리기 구간 (게임 카드 순서, 행동 번호, 길이, 일차): " +
                  ", ".join(f"{R['games'][g]['base']['day']}→{R['games'][g]['search']['day']}일 카드 행동 {k + 1}~{k + ln} ({d}일)" for g, k, ln, d in streaks[:6]))


if __name__ == "__main__":
    main()
