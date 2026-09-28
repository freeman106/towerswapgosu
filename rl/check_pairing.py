"""짝 비교 확인: 같은 게임들을 두 번 두되 두 번째에는 게임 0만 탐색을 끈다(개입).
나머지 게임의 행동·탐색 후보·Q·최종 일차가 모두 같아야 한다(한 게임의 개입이 다른 게임의 후보를 바꾸지 않음).
사용: python rl/check_pairing.py rl/runs/long1/agent_40M.pt [--games 32]"""
import argparse
import time

import numpy as np
import torch

from search_eval import Policy, play


class NoSearchFor:
    def __init__(self, games):
        self.games = games

    def before(self, env, alive):
        nosearch = np.zeros(len(alive), bool)
        nosearch[self.games] = True
        return {}, nosearch

    def after(self, env, i, a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    n = args.games
    runs = {}
    for name, ctl in (("기본", None), ("게임 0 탐색 끔", NoSearchFor([0]))):
        t0 = time.time()
        fr = [[] for _ in range(n)]
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        fin = play(pol, n, args.seed, True, 3, 2, 8, st, 2.0, frames=fr, ctl=ctl, no_toss=True)
        runs[name] = (fin, fr)
        print(f"{name}: day {fin.mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
    (fa, ra), (fb, rb) = runs.values()
    key = lambda f: (f["a"], f.get("base"), tuple(map(tuple, f.get("cand", []))))
    first_div = lambda x, y: next((k for k, (u, v) in enumerate(zip(x, y)) if key(u) != key(v)), None if len(x) == len(y) else min(len(x), len(y)))
    same = [i for i in range(1, n) if fa[i] == fb[i] and first_div(ra[i], rb[i]) is None]
    print(f"게임 0: 최종 {fa[0]} → {fb[0]}일, 첫 차이 {first_div(ra[0], rb[0])}번째 행동 (개입 대상이라 달라야 함)")
    print(f"게임 1~{n - 1}: 행동·후보·Q·최종 일차가 모두 같은 게임 {len(same)}/{n - 1}")
    for i in range(1, n):
        if i not in same:
            print(f"  다른 게임 {i}: 최종 {fa[i]} vs {fb[i]}, 첫 차이 {first_div(ra[i], rb[i])}번째 행동")


if __name__ == "__main__":
    main()
