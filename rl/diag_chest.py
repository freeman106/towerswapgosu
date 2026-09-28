"""상자 제한 진단: 늘어난 개봉 스왑이 왜 생존으로 이어지지 않는지 본다. 정책 혼자(search_eval 기본과 같은 잡음) 1,024판.
1. 일차 구간(1~3, 4~6, 7~10, 11~15, 16~20)별: 개봉 수입, 상자 관련 드래그 비용과 순수입, 하트(아침·밤 직전), 낮 하트 회복, 밤 피해, 밤 직전 방어물
2. 일반 상자 보유: 행동 단위 시간 평균과 밤 직전 평균, 5~6행에 있는 비율
3. 실제로 쓴 스왑(휴식일 제외 드래그)의 사용처
4. 10일·20일 보스 직전(밤 시작 직전) 하트와 방어 배치
구간 값은 그 구간에 살아 있던 게임-날 기준이다(구간마다 살아 있는 게임 집단이 다를 수 있어 게임 수도 함께 보인다).
사용: python rl/diag_chest.py ckpt [--no-open-normal] [--games 1024]"""
import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from search_eval import DIRS, Policy, act_cat, play

CHEST_SWAPS = {1: 2, 2: 12, 3: 70, 4: 380}
BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 20))
SPEND = ("상자 이동·합성", "자원 교환", "타워 이동·합성", "빈칸 이동", "버리기", "성 보수", "기타")


def bucket(day):
    return next((b for b in BUCKETS if b[0] <= day <= b[1]), None)


def drag_cat(cells, a):
    cat = act_cat(cells, 0, a)
    if cat == "버리기":
        return cat
    c = a // 4
    x, y = c % 6 + 1, c // 6 + 1
    dx, dy = DIRS[a % 4]
    tx, ty = x + dx, y + dy
    dst = cells[ty - 1][tx - 1] if 1 <= ty <= 7 and 1 <= tx <= 6 else ""
    if cells[y - 1][x - 1][:1] == "h" or dst[:1] == "h":
        return "상자 이동·합성"
    if cat == "돌 성에 넣기":
        return "성 보수"
    if cat in ("버리기", "빈칸 이동"):
        return cat
    if cat == "교환 자원-자원":
        return "자원 교환"
    if cat in ("교환 자원-타워", "교환 타워-타워"):
        return "타워 이동·합성"
    return "기타"


def layout(cells, turrets):
    """방어 배치 요약: 공격 타워 등급별 수(포탑 포함), 전체 방어물, 5~6행 방어물, 땅 점유율, 일반 상자 수, 5~6행 일반 상자"""
    atk = [0] * 5
    for c in [c for row in cells for c in row] + list(turrets):
        if c[:1] in ("t", "b", "c"):
            atk[min(int(c[1]), 4)] += 1
    flat = [(r + 1, c) for r, row in enumerate(cells) for c in row]
    towers = sum(1 for _, c in flat if c[:1] in "tbcw" and c)
    bottom = sum(1 for r, c in flat if r in (5, 6) and c[:1] in "tbcw" and c)
    land = sum(1 for _, c in flat if c != "~~")
    filled = sum(1 for _, c in flat if c not in ("~~", ".."))
    normal = sum(1 for _, c in flat if c == "h1")
    normal_bottom = sum(1 for r, c in flat if r in (5, 6) and c == "h1")
    return atk, towers, bottom, filled / land, normal, normal_bottom


class Diag:
    def __init__(self, n):
        self.day = np.zeros(n, int)
        self.morning = np.zeros(n, int)
        self.last = np.zeros(n, int)
        self.nights = []                   # (게임, 일차, 아침 하트, 밤 직전 하트, 다음 아침 하트)
        self.opens = []                    # (게임, 일차, 등급)
        self.spend = []                    # (게임, 일차, 사용처)
        self.dusk = {}                     # (게임, 일차) → 배치 요약 + 하트
        self.held = defaultdict(lambda: [0, 0, 0])  # 구간 → [일반 상자 수 합, 그중 5~6행, 행동 수]

    def before(self, env, alive):
        for i in np.nonzero(alive)[0]:
            day, hearts = env.state(int(i))[:2]
            if day != self.day[i]:
                if self.day[i]:
                    self.nights.append((int(i), int(self.day[i]), int(self.morning[i]), int(self.last[i]), hearts))
                self.day[i], self.morning[i] = day, hearts
            self.last[i] = hearts
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        day, hearts, _, phase, _ = env.state(int(i))
        if phase not in (0, 25):
            return
        cells, tur = env.board(int(i))
        b = bucket(day)
        if b:
            flat = [(r + 1, c) for r, row in enumerate(cells) for c in row]
            h = self.held[b]
            h[0] += sum(1 for _, c in flat if c == "h1")
            h[1] += sum(1 for r, c in flat if r in (5, 6) and c == "h1")
            h[2] += 1
        off = day % 10 == 1 and env.made(int(i))[2] == day
        if a < 168 and phase == 0 and not off:
            self.spend.append((int(i), day, drag_cat(cells, a)))
        elif 168 <= a < 216 and (a - 168) // 6 >= 1:
            y, x = (a - 168) // 6, (a - 168) % 6 + 1
            if cells[y - 1][x - 1][:1] == "h":
                self.opens.append((int(i), day, int(cells[y - 1][x - 1][1])))
        elif a == 224:
            self.dusk[(int(i), day)] = (hearts,) + layout(cells, tur)

    def finish(self):
        for i in np.nonzero(self.day)[0]:
            self.nights.append((int(i), int(self.day[i]), int(self.morning[i]), int(self.last[i]), 0))


def report(name, fin, d, n):
    print(f"\n=== {name}: day {fin.mean():.2f} · 10일 보스 통과 {np.mean(fin > 10):.1%} · 20일 보스 통과 {np.mean(fin > 20):.1%} ===")
    cols = [f"{lo}~{hi}일" for lo, hi in BUCKETS]
    rows = defaultdict(list)
    for lo, hi in BUCKETS:
        gd = sum(max(0, min(f, hi) - lo + 1) for f in fin)  # 살아 있던 게임-날
        alive = int(np.sum(fin >= lo))
        inc = sum(CHEST_SWAPS[t] for _, dd, t in d.opens if lo <= dd <= hi)
        sp = Counter(c for _, dd, c in d.spend if lo <= dd <= hi)
        ni = [x for x in d.nights if lo <= x[1] <= hi]
        dk = [v for (i, dd), v in d.dusk.items() if lo <= dd <= hi]
        rows["살아 있는 게임(구간 시작)"].append(f"{alive}")
        rows["개봉 수입/일"].append(f"{inc / gd:.2f}")
        rows["상자 드래그 비용/일"].append(f"{sp['상자 이동·합성'] / gd:.2f}")
        rows["순수입/일"].append(f"{(inc - sp['상자 이동·합성']) / gd:.2f}")
        rows["아침 하트"].append(f"{np.mean([x[2] for x in ni]):.1f}")
        rows["밤 직전 하트"].append(f"{np.mean([x[3] for x in ni]):.1f}")
        rows["낮 하트 회복/일"].append(f"{np.mean([x[3] - x[2] for x in ni]):+.2f}")
        rows["밤 피해/밤"].append(f"{np.mean([x[3] - x[4] for x in ni]):.2f}")
        rows["밤 직전 공격 타워 기본/동/은"].append("/".join(f"{np.mean([v[1][t] for v in dk]):.1f}" for t in (1, 2, 3)))
        rows["밤 직전 방어물 (5~6행)"].append(f"{np.mean([v[2] for v in dk]):.1f} ({np.mean([v[3] for v in dk]):.1f})")
        rows["밤 직전 점유율"].append(f"{np.mean([v[4] for v in dk]):.0%}")
        h = d.held[(lo, hi)]
        rows["일반 상자: 시간 평균 (5~6행)"].append(f"{h[0] / max(h[2], 1):.2f} ({h[1] / max(h[2], 1):.2f})")
        rows["일반 상자: 밤 직전 (5~6행)"].append(f"{np.mean([v[5] for v in dk]):.2f} ({np.mean([v[6] for v in dk]):.2f})")
        rows["최대 일반 상자(밤 직전)"].append(f"{max([v[5] for v in dk], default=0)}")
        tot = sum(sp.values())
        rows["쓴 스왑/일"].append(f"{tot / gd:.2f}")
        for c in SPEND:
            rows[f"  {c}"].append(f"{sp[c] / gd:.2f} ({sp[c] / max(tot, 1):.0%})")
    w = max(len(k) for k in rows) + 2
    print(" " * w + "".join(f"{c:>18s}" for c in cols))
    for k, v in rows.items():
        print(f"{k:<{w}s}" + "".join(f"{x:>18s}" for x in v))
    for bd in (10, 20):
        v = [v for (i, dd), v in d.dusk.items() if dd == bd]
        if v:
            print(f"  {bd}일 보스 직전 ({len(v)}판): 하트 {np.mean([x[0] for x in v]):.1f} · 공격 타워 기본/동/은 "
                  + "/".join(f"{np.mean([x[1][t] for x in v]):.1f}" for t in (1, 2, 3))
                  + f" · 방어물 {np.mean([x[2] for x in v]):.1f} (5~6행 {np.mean([x[3] for x in v]):.1f}) · 점유율 {np.mean([x[4] for x in v]):.0%}"
                  + f" · 일반 상자 {np.mean([x[5] for x in v]):.2f} (5~6행 {np.mean([x[6] for x in v]):.2f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-open-normal", action="store_true")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    t0 = time.time()
    d = Diag(args.games)
    fin = play(pol, args.games, args.seed, False, 0, 0, 0, None, ctl=d, no_toss=True, no_open=args.no_open_normal)
    d.finish()
    report(f"{args.ckpt} ({'개봉 제한' if args.no_open_normal else '제한 없음'}, {time.time() - t0:.0f}s)", fin, d, args.games)


if __name__ == "__main__":
    main()
