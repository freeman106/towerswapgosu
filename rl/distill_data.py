"""교사 실행 기록 재생 → 증류 데이터. teacher.py 실행(같은 시드·판 수)의 결정 기록(교체한 결정 전부 + 두 수 계획의 실제 재확인 결과)만으로
실제 궤적을 비싼 굴림 없이 다시 둔다: 정책·샘플링 잡음·상자 합성 계획은 원래 실행과 같고, 교사가 바꾼 자리(게임, 행동 번호)에서만 기록된 행동을 둔다.
교사는 실제 게임 난수를 쓰지 않으므로 같은 행동열이면 같은 게임이 된다. 확인: 기록된 결정 시점의 보드, 게임별 최종 일차.
'유지' 상태는 교사 후보 생성(VecEnv.teacher_plans, 빠름)과 teacher.py와 같은 후보 정리로 다시 찾는다(굴림은 하지 않는다).
샘플(실제 관측·환경 마스크·실제로 둔 행동): 분류
  1 교사 한 수(교체) / 2 교사 두 수 계획의 준비(첫) 행동 / 3 실제 준비 뒤 관측에서 둔 실제 완성 행동 / 4 교사가 평가하고 기존 선택을 유지 /
  5 준비 뒤 재확인 실패 상태의 정책 행동(완성 정답 아님) / 0 교사 평가 없는 상태(일부만, 변화 관찰용)
사용: python rl/distill_data.py --tag s12345 --games 1024 --seed 12345 --dec runs/teacher1024_s12345_decisions.json --games-json runs/teacher1024_s12345_games.json"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from diag_bronze_flow import run
from eval_restrict import CONDS, Restrict
from force_silver_prep import SEEDS
from ppo import A, C, H, S, W, parse_emergency
from search_eval import Policy
from teacher import swap_key

CAT = {"그 밖": 0, "교사 한 수": 1, "교사 준비": 2, "실제 완성": 3, "유지": 4, "재확인 실패 뒤": 5}


class Replay(Restrict):
    def __init__(self, n, dec, other_frac, seed):
        super().__init__(n)
        self.rep = {(d["game"], d["step"]): d for d in dec if d["chosen"]}
        self.pending = [None] * n
        self.pstep = 0
        self.other_frac, self.rng = other_frac, np.random.default_rng([seed, 55])
        self.og = np.zeros((n, C, H, W), np.float32)
        self.os = np.zeros((n, S), np.float32)
        self.om = np.zeros((n, A), bool)
        self.out = {k: [] for k in ("grid", "scal", "mask", "act", "base", "cat", "game", "day", "step", "failed", "same")}
        self.check = Counter()

    def force(self, env, alive, logits, noise, act):
        forced = super().force(env, alive, logits, noise, act)  # 상자 합성 계획(기존과 같음)
        self.pstep += 1
        live = [int(i) for i in np.nonzero(alive)[0]]
        cat = {}
        recheck = []
        for i in live:
            pend = self.pending[i]
            if pend is None:
                continue
            c = self.cur[i]
            if i in forced or c["day"] != pend["day"]:
                assert pend["done"] is None, f"게임 {i}: 원래 실행은 계획을 끝까지 확인했는데 재생에서는 중단"
                self.pending[i] = None
                self.check["두 수 계획 중단"] += 1
            elif c["phase"] == 0:
                recheck.append(i)
        for i in recheck:
            pend = self.pending[i]
            self.pending[i] = None
            assert pend["done"] is not None, f"게임 {i}: 원래 실행은 중단했는데 재생에서는 재확인 상태"
            if pend["done"]:
                if act[i] != pend["a2"]:
                    forced[i] = pend["a2"]
                cat[i] = ("실제 완성", pend)
            else:
                cat[i] = ("재확인 실패 뒤", pend)
        cand = [i for i in live if i not in forced and i not in recheck and self.pending[i] is None
                and self.cur[i]["phase"] == 0 and self.cur[i]["swaps"] >= 1
                and not (self.cur[i]["day"] % 10 == 1 and env.made(i)[2] == self.cur[i]["day"])]
        if cand:
            P = env.teacher_plans(np.array(cand, np.int64), SEEDS)
            for i, L in zip(cand, P):
                base = int(act[i])
                has = any(not (p[1] < 0 and swap_key(p[0]) == swap_key(base)) for p in L)
                d = self.rep.get((i, int(self.step[i])))
                if d is None:
                    if has:
                        cat[i] = ("유지", None)
                    continue
                assert has, f"게임 {i} 행동 {self.step[i]}: 기록된 교체 결정인데 재생에서는 교사 후보가 없음"
                board = [x for row in env.board(i)[0] for x in row]
                assert board == d["board"], f"게임 {i} 행동 {self.step[i]}: 보드가 기록과 다름"
                self.check["결정 보드 일치"] += 1
                a1, a2 = d["chosen"]
                if a1 != base:
                    forced[i] = a1
                if a2 >= 0:
                    self.pending[i] = {"a2": a2, "done": d.get("완성"), "day": self.cur[i]["day"]}
                    cat[i] = ("교사 준비", {"failed": d.get("완성") is False, "same": a1 == base})
                else:
                    cat[i] = ("교사 한 수", {"failed": False, "same": False})
        env.observe(self.og, self.os, self.om)
        for i in live:
            name, info = cat.get(i, ("그 밖", None))
            if name == "그 밖" and self.rng.random() >= self.other_frac:
                continue
            o = self.out
            o["grid"].append(self.og[i].astype(np.float16))
            o["scal"].append(self.os[i].astype(np.float16))
            o["mask"].append(np.packbits(self.om[i]))
            o["act"].append(int(forced.get(i, act[i])))
            o["base"].append(int(act[i]))
            o["cat"].append(CAT[name])
            o["game"].append(i)
            o["day"].append(self.cur[i]["day"])
            o["step"].append(int(self.step[i]))
            o["failed"].append(bool(info and info.get("failed")) if name == "교사 준비" else False)
            o["same"].append(bool(info and info.get("same")) if name == "교사 준비" else False)
        return forced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--tag", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dec", required=True)
    p.add_argument("--games-json", required=True)
    p.add_argument("--other-frac", type=float, default=0.05)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    dec = json.load(open(args.dec))
    want = np.array(json.load(open(args.games_json))["T"]["final"])
    t0 = time.time()
    c = Replay(args.games, dec, args.other_frac, args.seed)
    fin, _, _ = run(pol, args.games, args.seed, rule, None, ctl=c)
    same = fin == want
    print(f"재생 {time.time() - t0:.0f}s · 최종 일차 일치 {same.sum()}/{len(same)} · 확인 {dict(c.check)}")
    assert same.all(), f"최종 일차가 다른 게임 {np.nonzero(~same)[0][:10]}"
    o = c.out
    arr = {k: np.array(v) for k, v in o.items()}
    names = {v: k for k, v in CAT.items()}
    cnt = Counter(arr["cat"].tolist())
    print("샘플:", {names[k]: v for k, v in sorted(cnt.items())}, "· 교사 준비 중 재확인 실패", int(arr["failed"].sum()),
          "· 준비 첫 수가 정책 선택과 같음", int(arr["same"].sum()))
    np.savez_compressed(f"runs/distill_{args.tag}.npz", seed=args.seed, **arr)
    print(f"저장 runs/distill_{args.tag}.npz")


if __name__ == "__main__":
    main()
