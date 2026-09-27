"""사망 원인 분석: 정책으로 게임을 끝까지 두고 요약한다.
- 사망 일차 분포와 밤별 사망 위험(그 밤까지 온 게임 중 그 밤에 죽은 비율)
- 보스(10·20·30·40·50일) 도달률과 도달 후 통과율
- 밤마다 잃은 하트, 사망 직전 3일의 하트
- 마지막 밤 직전 보드: 방어 배치(종류·등급), 남은 상자. 같은 날 밤을 넘긴 게임과 비교
사용: python rl/diag_death.py rl/runs/cb9/agent.pt [--games 1024] [--envs 512] [--boards 3]"""
import argparse
from collections import defaultdict

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent, masked_dist

A_END_DAY = 224
BOSS_DAYS = (10, 20, 30, 40, 50)
TOWER_KINDS = {"t": "화살탑", "b": "발리스타", "c": "대포", "w": "얼음벽", "I": "빙산"}
OPEN_SWAPS = {1: 2, 2: 12, 3: 70, 4: 380}


def tile(s):
    """보드 문자열 칸 → (종류, 등급) 또는 None"""
    if s in ("..", "~~", ""):
        return None
    return s[0], int(s[1]) if s[1].isdigit() else 0


def summarize_board(cells, turrets):
    """방어 가치(3^(등급-1), 얼음벽·빙산은 절반), 종류별 가치, 등급별 타워 수, 남은 상자의 개봉 스왑"""
    kinds = defaultdict(float)
    tiers = defaultdict(int)
    chests = 0
    for s in [c for row in cells for c in row] + list(turrets):
        t = tile(s)
        if t is None:
            continue
        k, tier = t
        if k in TOWER_KINDS:
            v = 3 ** (max(tier, 1) - 1) * (0.5 if k in "wI" else 1.0)
            kinds[k] += v
            if k in "tbc":
                tiers[tier] += 1
        elif k == "h":
            chests += OPEN_SWAPS.get(tier, 0)
    return {"defense": sum(kinds.values()), "kinds": dict(kinds), "tiers": dict(tiers), "chests": chests}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--boards", type=int, default=3, help="출력할 대표 실패 보드 수")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
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
    env.reset(grid, scal, mask)

    # 게임마다: 밤 직전(하루 끝내기 직전) 하트·보드, 다음 날 아침 하트
    new_game = lambda: {"dusk": {}, "morning": {}, "boards": {}}
    cur = [new_game() for _ in range(N)]
    k = -(-args.games // N)
    per = [[] for _ in range(N)]
    while min(len(x) for x in per) < k:
        with torch.no_grad():
            logits, _ = agent(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            act = masked_dist(logits, torch.from_numpy(mask).to(dev)).sample().cpu().numpy()
        for i in np.nonzero(act == A_END_DAY)[0]:
            day, hearts, _, _, _ = env.state(int(i))
            cells, turrets = env.board(int(i))
            cur[i]["dusk"][day] = hearts
            cur[i]["boards"][day] = (cells, turrets)
        env.step(act, grid, scal, mask, rew, done, days)
        fin = {f["env"]: f for f in env.pop_finished()}
        for i in np.nonzero(days > 0)[0]:
            if not done[i]:
                day, hearts, _, _, _ = env.state(int(i))
                cur[i]["morning"][day] = hearts
        for i in np.nonzero(done)[0]:
            f = fin[int(i)]
            if len(per[i]) < k and not f["truncated"]:
                g = cur[i]
                g.update(day=f["day"], score=f["score"])
                per[i].append(g)
            cur[i] = new_game()
    games = [g for x in per for g in x]
    dd = np.array([g["day"] for g in games])
    n = len(games)
    print(f"{args.ckpt}: 게임 {n} · day 평균 {dd.mean():.2f} · 중앙 {int(np.median(dd))} · 최대 {dd.max()} · "
          f"점수 평균 {np.mean([g['score'] for g in games]):.0f}")

    # 1. 사망 일차 분포 (5일 구간)와 밤별 위험
    print("\n[사망 일차 분포]")
    for lo in range(1, dd.max() + 1, 5):
        c = ((dd >= lo) & (dd < lo + 5)).sum()
        print(f"  {lo:2d}~{lo + 4:2d}일: {c:4d} ({c / n:5.1%}) {'#' * int(60 * c / n)}")
    print("\n[밤별 사망 위험] 그 밤까지 온 게임 중 그 밤에 죽은 비율 (보스 밤 *)")
    line = []
    for d in range(1, dd.max() + 1):
        reach = (dd >= d).sum()
        if reach < 20:
            break
        line.append(f"{d}{'*' if d in BOSS_DAYS else ''}:{(dd == d).sum() / reach:.0%}")
    for j in range(0, len(line), 10):
        print("  " + "  ".join(line[j:j + 10]))

    # 2. 보스
    print("\n[보스] 도달(그날 밤까지 생존) · 도달 후 통과 · 직전 3밤 평균 사망 위험")
    for b in BOSS_DAYS:
        reach = (dd >= b).mean()
        if reach == 0:
            break
        passed = (dd > b).sum() / max((dd >= b).sum(), 1)
        pre = np.mean([(dd == d).sum() / max((dd >= d).sum(), 1) for d in range(b - 3, b)])
        print(f"  {b}일: 도달 {reach:5.1%} · 통과 {passed:5.1%} (보스 밤 사망 {1 - passed:5.1%} vs 직전 3밤 평균 {pre:5.1%})")
    for d in (30, 40, 50):
        print(f"  {d}일 도달(그날 아침까지 생존): {(dd >= d).mean():.1%}")

    # 3. 밤마다 잃은 하트 (그 밤 직전 하트 - 다음 날 아침 하트, 죽은 밤은 직전 하트 전부)
    loss = defaultdict(list)
    for g in games:
        for d, h in g["dusk"].items():
            after = 0 if d == g["day"] else g["morning"].get(d + 1)
            if after is not None:
                loss[d].append(h - after)
    print("\n[밤별 잃은 하트] 평균 (1개 이상 잃은 비율)")
    line = [f"{d}{'*' if d in BOSS_DAYS else ''}:{np.mean(v):.2f}({np.mean(np.array(v) > 0):.0%})"
            for d, v in sorted(loss.items()) if len(v) >= 20]
    for j in range(0, len(line), 8):
        print("  " + "  ".join(line[j:j + 8]))

    # 4. 사망 직전 3일의 하트 (밤 직전 하트), 보스 밤 사망과 일반 밤 사망 구분
    print("\n[사망 직전 하트] 밤 직전 하트: D-3, D-2, D-1, 사망한 밤 D")
    for name, sel in (("보스 밤 사망", lambda g: g["day"] in BOSS_DAYS), ("일반 밤 사망", lambda g: g["day"] not in BOSS_DAYS)):
        gs = [g for g in games if sel(g) and all(g["day"] - j in g["dusk"] for j in range(4))]
        if gs:
            hs = np.array([[g["dusk"][g["day"] - j] for j in (3, 2, 1, 0)] for g in gs])
            print(f"  {name} {len(gs)}게임: " + " → ".join(f"{x:.2f}" for x in hs.mean(0)) +
                  f" · 마지막 밤 직전 하트 5개 이상 {np.mean(hs[:, 3] >= 5):.0%}, 1개 {np.mean(hs[:, 3] == 1):.0%}")

    # 5. 마지막 밤 직전 보드: 죽은 게임 vs 같은 날 밤을 넘긴 게임
    print("\n[밤 직전 보드] 같은 날 밤에 죽은 게임 vs 넘긴 게임 (방어 가치, 종류별, T1/동/은/금 타워 수, 남은 상자 개봉 스왑, 하트)")
    for lo, hi in ((15, 19), (20, 20), (21, 29), (30, 30), (31, 40)):
        rows = {True: [], False: []}
        for g in games:
            for d, (cells, tur) in g["boards"].items():
                if lo <= d <= hi:
                    s = summarize_board(cells, tur)
                    s["hearts"] = g["dusk"][d]
                    rows[d == g["day"]].append(s)
        if len(rows[True]) < 5:
            continue
        print(f"  {lo}~{hi}일" if lo != hi else f"  {lo}일")
        for died in (True, False):
            r = rows[died]
            kinds = " ".join(f"{TOWER_KINDS[k]} {np.mean([x['kinds'].get(k, 0) for x in r]):.1f}" for k in TOWER_KINDS)
            tiers = "/".join(f"{np.mean([x['tiers'].get(t, 0) for x in r]):.1f}" for t in (1, 2, 3, 4))
            print(f"    {'사망' if died else '생존'} {len(r):5d}: 방어 {np.mean([x['defense'] for x in r]):5.1f} ({kinds}) "
                  f"타워 {tiers} · 상자 {np.mean([x['chests'] for x in r]):.1f} · 하트 {np.mean([x['hearts'] for x in r]):.2f}")

    # 6. 대표 실패 보드: 20일 보스, 30일 보스, 중앙값 근처 일반 밤
    med = int(np.median(dd))
    picks = []
    for want in (20, 30, med if med not in BOSS_DAYS else med + 1):
        g = next((g for g in games if g["day"] == want and want in g["boards"]), None)
        if g:
            picks.append(g)
    for g in picks[:args.boards]:
        d = g["day"]
        cells, tur = g["boards"][d]
        print(f"\n[실패 보드] {d}일 밤 사망{' (보스)' if d in BOSS_DAYS else ''}, 밤 직전 하트 {g['dusk'][d]}, "
              f"하트 추이 " + " ".join(str(g["dusk"].get(x, "-")) for x in range(max(1, d - 5), d + 1)))
        print("  포탑: " + " ".join(t or "--" for t in tur))
        for y, row in enumerate(cells, 1):
            print(f"  {y}: " + " ".join(f"{c:3s}" for c in row))


if __name__ == "__main__":
    main()
