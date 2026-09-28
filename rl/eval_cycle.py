"""은상자 순환 평가 (정책 혼자, search_eval 기본과 같은 판·스텝별 잡음). 모델마다 상자 개봉 규칙을 지정한다.
- 최종 성능: 점수, 생존일, 20·30일 보스 통과, 35·40일 도달
- 경제: 등급별 생성·개봉, 비상 개봉, 개봉 수입과 상자 이동·합성 드래그 비용
- 은상자 깔때기: 첫 은상자 생성·첫 개봉·두 번째 개봉 비율과 시점, 첫→둘째 개봉 사이 일수와 실제로 쓴 스왑(사용처별)
- 필드 성장(첫 은상자를 연 게임, 도중에 죽은 판 포함): 개봉 직전·그날 밤 직전·이후 3일 밤 직전의 공격 타워 등급별 수와
  5~6행 위치, 은 이상 무기, 기본 환산 전력(Σ 3^(등급-1)), 채운 칸, 하트, 엔진 통계로 센 새 무기 생성(이동은 세지 않음)
- 스왑 사용처: 첫 은상자 개봉 전 3일 대 개봉일 포함 4일 (하루당)
- 게임 분류: 은상자 없음 / 만들었지만 못 엶 / 열었지만 필드 강화 못 함 / 강화했지만 다음 은상자 없음 / 두 번째 은상자 개봉
  (강화 = 개봉일부터 3일 안에 은 이상 공격 타워가 엔진 통계로 1개 이상 새로 생김)
사용: python rl/eval_cycle.py ckpt [--open-min-tier 3 --emergency 2,0,5] [--games 256 --seed 777] [--out 결과.json]"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_chest import drag_cat
from ppo import parse_emergency
from search_eval import Policy, play

CHEST_SWAPS = {1: 2, 2: 12, 3: 70, 4: 380}
SPEND = ("상자 이동·합성", "자원 교환", "타워 이동·합성", "성 보수", "버리기", "빈칸 이동", "기타")
SNAPS = ("개봉 직전", "당일 밤 직전", "+1일 밤 직전", "+2일 밤 직전", "+3일 밤 직전")
RULE_MIN = 1  # 평가 중인 개봉 규칙의 최소 등급 (그보다 낮은 등급 개봉 = 비상 개봉)
SILVER = 3  # 분석 기준 등급 (--silver-tier로 바꿀 수 있다, 시험용)
CLASSES = ("은상자 없음", "만들었지만 못 엶", "열었지만 강화 못 함", "강화했지만 다음 은상자 없음", "두 번째 은상자 개봉")


def snapshot(env, i, cells, turrets, hearts):
    atk, bottom = [0] * 5, [0] * 5
    for r, row in enumerate(cells, start=1):
        for c in row:
            if c[:1] in ("t", "b", "c"):
                t = min(int(c[1]), 4)
                atk[t] += 1
                bottom[t] += r in (5, 6)
    for c in turrets:
        if c[:1] == "t":
            atk[min(int(c[1]), 4)] += 1
    made = env.made(int(i))[0]  # 종류(화살탑·발리스타·대포·얼음벽·상자) × 등급 1..4 누적 생성
    return {"atk": atk[1:], "bottom": bottom[1:], "strength": sum(atk[t] * 3 ** (t - 1) for t in range(1, 5)),
            "filled": sum(1 for row in cells for c in row if c not in ("~~", "..")), "hearts": hearts,
            "made_w": [sum(made[k][t] for k in range(3)) for t in range(4)]}


class Cycle:
    """play()의 ctl: 상자 개봉, 드래그 사용처, 첫 은상자 생성일, 첫 은상자 개봉 전후 필드 스냅숏을 기록한다"""

    def __init__(self, n):
        self.step = np.zeros(n, int)
        self.silver_made = np.full(n, -1)
        self.first_open = np.full(n, -1)
        self.opens = [[] for _ in range(n)]   # (스텝, 일차, 등급)
        self.drags = [[] for _ in range(n)]   # (스텝, 일차, 사용처)
        self.snap = [{} for _ in range(n)]

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        i = int(i)
        day, hearts, _, phase, _ = env.state(i)
        st = self.step[i]
        self.step[i] += 1
        if phase not in (0, 25):
            return
        cells, tur = env.board(i)
        if self.silver_made[i] < 0 and any(c[:1] == "h" and int(c[1]) >= SILVER for row in cells for c in row):
            self.silver_made[i] = day
        if a < 168:
            if phase == 0 and not (day % 10 == 1 and env.made(i)[2] == day):
                self.drags[i].append((int(st), day, drag_cat(cells, a)))
        elif a < 216:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if y >= 1 and cells[y - 1][x - 1][:1] == "h":
                tier = int(cells[y - 1][x - 1][1])
                self.opens[i].append((int(st), day, tier))
                if tier >= SILVER and self.first_open[i] < 0:
                    self.first_open[i] = day
                    self.snap[i][SNAPS[0]] = snapshot(env, i, cells, tur, hearts)
        elif a == 224 and self.first_open[i] >= 0 and 0 <= day - self.first_open[i] <= 3:
            self.snap[i][SNAPS[1 + day - self.first_open[i]]] = snapshot(env, i, cells, tur, hearts)


def analyze(fin, fins, c, n):
    """게임별 요약과 분류"""
    games = []
    for i in range(n):
        f = fins[i]
        silver_opens = [(s, d) for s, d, t in c.opens[i] if t >= SILVER]
        g = {"game": i, "final": int(fin[i]), "silver_made": int(c.silver_made[i]), "first_open": int(c.first_open[i]),
             "second_open": silver_opens[1][1] if len(silver_opens) > 1 else -1,
             "made": f["made"], "opened": f["opened"], "emergency": f["emergency_opens"]}
        if len(silver_opens) > 1:
            s1, s2 = silver_opens[0][0], silver_opens[1][0]
            g["between_days"] = silver_opens[1][1] - silver_opens[0][1]
            g["between_spend"] = dict(Counter(cat for s, _, cat in c.drags[i] if s1 <= s < s2))
        sn = c.snap[i]
        if g["first_open"] >= 0:
            last = next(sn[k] for k in reversed(SNAPS) if k in sn)
            b = sn[SNAPS[0]]
            g["new_silver_w"] = sum(last["made_w"][2:]) - sum(b["made_w"][2:])
            g["d_strength"] = last["strength"] - b["strength"]
            g["died_within3"] = bool(fin[i] < g["first_open"] + 3)
            g["strengthened"] = g["new_silver_w"] >= 1
        if g["silver_made"] < 0:
            g["class"] = CLASSES[0]
        elif g["first_open"] < 0:
            g["class"] = CLASSES[1]
        elif g["second_open"] >= 0:
            g["class"] = CLASSES[4]
        elif not g["strengthened"]:
            g["class"] = CLASSES[2]
        else:
            g["class"] = CLASSES[3]
        games.append(g)
    return games


def report(title, fin, fins, c, games, n):
    score = np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin])
    made, opened = np.array([f["made"] for f in fins]), np.array([f["opened"] for f in fins])
    emerg = np.array([f["emergency_opens"] for f in fins])
    print(f"\n=== {title} ===")
    print(f"  성능: 점수 {score:.0f} · day {fin.mean():.2f} (중앙 {int(np.median(fin))}) · 20일 보스 통과 {np.mean(fin > 20):.1%} · "
          f"30일 보스 통과 {np.mean(fin > 30):.1%} · 35일 도달 {np.mean(fin >= 35):.1%} · 40일 도달 {np.mean(fin >= 40):.1%}")
    print(f"  상자 생성/개봉 (게임당, 일반·동·은·금): {' · '.join(f'{m:.2f}/{o:.2f}' for m, o in zip(made.mean(0), opened.mean(0)))} | "
          f"비상 동상자 개봉 {emerg.mean():.2f}/게임 (쓴 게임 {np.mean(emerg > 0):.0%})")
    opps, taken = sum(f["merge_opps"] for f in fins), sum(f["merge_taken"] for f in fins)
    print(f"  상자 합성 기회(교환 한 번으로 합성 가능한 입력 대기 상태) {opps / n:.2f}/게임 · 그중 합성을 고른 비율 {taken / max(opps, 1):.0%}")
    if RULE_MIN > 2:
        em_days = np.array([d for o in c.opens for _, d, t in o if t < RULE_MIN])
        if len(em_days):
            print("  비상 개봉 일차 분포: " + " · ".join(
                f"{lo}~{hi if hi < 99 else ''}일 {np.mean((em_days >= lo) & (em_days <= hi)):.0%}" for lo, hi in ((1, 3), (4, 6), (7, 10), (11, 20), (21, 99)))
                + f" (하트 5 이하 조건은 시작 하트가 5라 초반에 자주 맞는다)")
    gd = fin.sum()
    inc = sum(CHEST_SWAPS[t] for o in c.opens for _, _, t in o)
    chest_drag = sum(1 for d in c.drags for _, _, cat in d if cat == SPEND[0])
    print(f"  개봉 수입 {inc / gd:.2f}/게임-날 · 상자 이동·합성 드래그 {chest_drag / gd:.2f}/게임-날 · 순수입 {(inc - chest_drag) / gd:.2f}")
    sm, fo, so = (np.array([g[k] for g in games]) for k in ("silver_made", "first_open", "second_open"))
    q = lambda x: f"평균 {x.mean():.1f} · 중앙 {np.median(x):.0f}" if len(x) else "-"
    print(f"  은상자 깔때기: 첫 생성 {np.mean(sm >= 0):.1%} ({q(sm[sm >= 0])}일) → 첫 개봉 {np.mean(fo >= 0):.1%} ({q(fo[fo >= 0])}일) → "
          f"두 번째 개봉 {np.mean(so >= 0):.1%} ({q(so[so >= 0])}일)")
    two = [g for g in games if "between_days" in g]
    if two:
        sp = Counter()
        for g in two:
            sp.update(g["between_spend"])
        tot = sum(sp.values())
        print(f"  첫→둘째 은상자 개봉: {np.mean([g['between_days'] for g in two]):.1f}일 · 쓴 스왑 {tot / len(two):.1f}/게임 ("
              + ", ".join(f"{k} {sp[k] / len(two):.1f}" for k in SPEND if sp[k]) + ")")
    op = [i for i in range(n) if c.first_open[i] >= 0]
    if op:
        print(f"  필드 성장 (첫 은상자를 연 {len(op)}판, 3일 안에 죽은 판 {np.mean([games[i]['died_within3'] for i in op]):.0%} 포함):")
        print(f"    {'시점':<12s}{'판':>5s}{'공격 타워 기본/동/은/금':>22s}{'5~6행 기본/동/은/금':>22s}{'은+ 무기':>9s}{'전력':>8s}{'채운 칸':>8s}{'하트':>7s}"
              f"{'새 무기 생성 기본/동/은/금(개봉 직전부터)':>34s}")
        for k in SNAPS:
            v = [c.snap[i][k] for i in op if k in c.snap[i]]
            if not v:
                continue
            b = [c.snap[i][SNAPS[0]] for i in op if k in c.snap[i]]
            m = lambda key, t=None: np.mean([x[key][t] if t is not None else x[key] for x in v])
            dm = "/".join(f"{np.mean([x['made_w'][t] - y['made_w'][t] for x, y in zip(v, b)]):.1f}" for t in range(4))
            atk = "/".join(f"{m('atk', t):.1f}" for t in range(4))
            bot = "/".join(f"{m('bottom', t):.1f}" for t in range(4))
            print(f"    {k:<12s}{len(v):>5d}{atk:>22s}{bot:>22s}{np.mean([sum(x['atk'][2:]) for x in v]):>9.1f}"
                  f"{m('strength'):>8.1f}{m('filled'):>8.1f}{m('hearts'):>7.1f}{dm:>34s}")
        before, after = Counter(), Counter()
        nb = na = 0
        for i in op:
            d0 = c.first_open[i]
            before.update(cat for _, d, cat in c.drags[i] if d0 - 3 <= d < d0)
            after.update(cat for _, d, cat in c.drags[i] if d0 <= d <= d0 + 3)
            nb += max(0, min(3, d0 - 1))
            na += min(4, fin[i] - d0 + 1)
        print("    스왑 사용처 (하루당, 개봉 전 3일 → 개봉일 포함 4일): "
              + ", ".join(f"{k} {before[k] / max(nb, 1):.2f}→{after[k] / max(na, 1):.2f}" for k in SPEND if before[k] or after[k]))
    cl = Counter(g["class"] for g in games)
    print("  게임 분류: " + " | ".join(
        f"{k} {cl[k] / n:.1%} (평균 {np.mean([g['final'] for g in games if g['class'] == k]):.1f}일)" for k in CLASSES if cl[k]))
    opened_g = [g for g in games if g["first_open"] >= 0]
    if opened_g:
        print(f"    첫 은상자 개봉 판: 3일 안 사망 {np.mean([g['died_within3'] for g in opened_g]):.0%} · 은+ 무기 새로 생성 평균 "
              f"{np.mean([g['new_silver_w'] for g in opened_g]):.2f} · 전력 변화 평균 {np.mean([g['d_strength'] for g in opened_g]):+.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--open-min-tier", type=int, default=1)
    p.add_argument("--emergency", default="")
    p.add_argument("--toss-ban", type=int, default=1, help="휴식일 버리기 금지 (기본 켬)")
    p.add_argument("--out", default="")
    p.add_argument("--silver-tier", type=int, default=3, help="분석 기준 상자 등급 (기본 은상자 3, 시험용으로만 바꾼다)")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    global SILVER, RULE_MIN
    SILVER, RULE_MIN = args.silver_tier, args.open_min_tier
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.games
    t0 = time.time()
    c, fins = Cycle(n), []
    rule = {"open_min_tier": args.open_min_tier, "emergency": parse_emergency(args.emergency)}
    fin = play(pol, n, args.seed, False, 0, 0, 0, None, ctl=c, no_toss=bool(args.toss_ban), fin_rec=fins, env_kw=rule)
    fins.sort(key=lambda f: f["env"])
    games = analyze(fin, fins, c, n)
    rule_s = "제한 없음" if args.open_min_tier <= 1 else f"{args.open_min_tier}등급부터 개봉" + (f", 비상 예외 {args.emergency}" if args.emergency else "")
    report(f"{args.ckpt} ({rule_s}, 시드 {args.seed} {n}판, {time.time() - t0:.0f}s)", fin, fins, c, games, n)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "rule": rule, "seed": args.seed, "games": games}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
