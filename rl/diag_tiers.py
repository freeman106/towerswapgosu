"""등급 합성 진단: 교환 한 번이면 같은 종류·등급 3개가 이어져 윗등급이 되는 상황(합성 기회)을 등급별로 세고,
그때 정책이 합성하는 교환을 골랐는지 본다. 타워(화살탑·발리스타·대포·얼음벽)와 상자를 따로 센다.
밤 직전 보드의 등급별 타워 수도 날짜 구간별로 낸다.
사용: python rl/diag_tiers.py rl/runs/cb9/agent.pt [--games 256]   /   python rl/diag_tiers.py greedy"""
import argparse
from collections import defaultdict

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent, masked_dist

TOWERS = "tbcw"
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 위·아래·왼·오 (행동 번호 순서)


def parse(cells):
    """보드 문자열 → {(x, y): (종류, 등급)} (x 1..6, y 1..7)"""
    b = {}
    for y, row in enumerate(cells, 1):
        for x, s in enumerate(row, 1):
            if s not in ("..", "~~", ""):
                b[(x, y)] = (s[0], int(s[1]) if s[1].isdigit() else 0)
    return b


def line3(b, x, y):
    """(x, y)를 지나는 같은 (종류, 등급) 줄이 3 이상이면 그 (종류, 등급)"""
    t = b.get((x, y))
    if t is None or t[1] >= 4 or t[0] not in TOWERS + "h":
        return None
    for dx, dy in ((1, 0), (0, 1)):
        n = 1
        for sgn in (1, -1):
            px, py = x + sgn * dx, y + sgn * dy
            while b.get((px, py)) == t:
                n += 1
                px, py = px + sgn * dx, py + sgn * dy
        if n >= 3:
            return t
    return None


def merge_actions(b):
    """교환 한 번으로 생기는 합성: {행동 번호: 결과 (종류, 새 등급)}. 이웃 두 타일의 교환만 본다(근사)"""
    out = {}
    for (x, y), ta in b.items():
        for d, (dx, dy) in enumerate(DIRS):
            q = (x + dx, y + dy)
            tb = b.get(q)
            if tb is None or tb == ta or "C" in (ta[0], tb[0]):
                continue
            nb = dict(b)
            nb[(x, y)], nb[q] = tb, ta
            for cx, cy in ((x, y), q):
                r = line3(nb, cx, cy)
                if r:
                    out[((y - 1) * 6 + (x - 1)) * 4 + d] = (r[0], r[1] + 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="체크포인트 경로 또는 greedy")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    greedy = args.policy == "greedy"
    if not greedy:
        dev = torch.device(args.device)
        ck = torch.load(args.policy, map_location=dev, weights_only=False)
        a = ck["args"]
        agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
        agent.load_state_dict(ck["agent"])
        agent.eval()

    N = args.envs
    env = ts.VecEnv(N, seed=args.seed)
    grid = np.zeros((N, C, H, W), np.float32)
    scal = np.zeros((N, S), np.float32)
    mask = np.zeros((N, A), bool)
    rew, done, days = np.zeros(N, np.float32), np.zeros(N, bool), np.zeros(N, np.float32)
    ex_act, ex_best = np.zeros(N, np.int64), np.zeros((N, A), bool)
    env.reset(grid, scal, mask)

    opp = defaultdict(int)    # (분류, 새 등급) → 합성 기회가 있던 결정 수
    take = defaultdict(int)   # 그중 합성 교환을 고른 수
    pmass = defaultdict(float)  # 정책이 합성 교환에 둔 확률 합 (PPO만)
    dusk = defaultdict(lambda: defaultdict(list))  # 날짜 구간 → 등급 → 타워 수 목록
    finished = 0
    while finished < args.games:
        if greedy:
            env.expert(ex_act, ex_best)
            act, probs = ex_act.copy(), None
        else:
            with torch.no_grad():
                logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
                dist = masked_dist(logits, torch.from_numpy(mask).to(dev))
                act = dist.sample().cpu().numpy()
                probs = dist.probs.cpu().numpy()
        for i in range(N):
            day, hearts, swaps, phase, _ = env.state(i)
            if phase == 25 and act[i] == 224:
                cells, _ = env.board(i)
                b = parse(cells)
                bucket = "1~9" if day < 10 else "10~19" if day < 20 else "20~29" if day < 30 else "30+"
                for t in (1, 2, 3, 4):
                    dusk[bucket][t].append(sum(1 for k, tt in b.values() if k in "tbc" and tt == t))
            if phase != 0 or swaps < 1:
                continue
            ma = merge_actions(parse(env.board(i)[0]))
            if not ma:
                continue
            groups = defaultdict(list)
            for a_, (k, t) in ma.items():
                if mask[i, a_]:
                    groups[("상자" if k == "h" else "타워", t)].append(a_)
            for g, acts in groups.items():
                opp[g] += 1
                take[g] += int(act[i] in acts)
                if probs is not None:
                    pmass[g] += float(probs[i, acts].sum())
        env.step(act, grid, scal, mask, rew, done, days)
        finished += len(env.pop_finished())

    name = "greedy" if greedy else args.policy
    print(f"{name}: 게임 {finished}")
    print("[교환 한 번 합성 기회] 결과 등급별: 게임당 기회(결정 수) · 합성 선택 비율" + ("" if greedy else " · 합성 교환에 둔 확률"))
    tn = ["", "", "동", "은", "금"]
    for cls in ("타워", "상자"):
        for t in (2, 3, 4):
            g = (cls, t)
            if opp[g]:
                extra = "" if greedy else f" · 확률 {pmass[g] / opp[g]:.1%}"
                print(f"  {cls} → {tn[t]}: 기회 {opp[g] / finished:.2f}/게임 · 선택 {take[g] / opp[g]:.1%}{extra}")
    print("[밤 직전 등급별 타워 수 (화살탑·발리스타·대포)] 기본/동/은/금")
    for bucket in ("1~9", "10~19", "20~29", "30+"):
        if dusk[bucket][1]:
            print(f"  {bucket}일: " + " / ".join(f"{np.mean(dusk[bucket][t]):.1f}" for t in (1, 2, 3, 4)))


if __name__ == "__main__":
    main()
