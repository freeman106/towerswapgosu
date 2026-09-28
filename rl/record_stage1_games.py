"""은상자 연습(1단계) 리플레이용 짝 기록: eval_stage1.py와 같은 평가 풀·시드에서 같은 시작 상태(은상자 배치, 같은 난수)를
모델 A와 B가 각각 둔 기록을 저장한다(build_replay.py 입력 형식, search 칸 = A, base 칸 = B).
먼저 프레임 없이 두어 사례를 고르고, 고른 게임만 프레임을 기록해 다시 둔다(같은 판 수·시드라 결과가 같다).
사용: python rl/record_stage1_games.py A.pt B.pt out.json [--name-a ... --name-b ...]"""
import argparse
import json

import numpy as np
import torch

from eval_stage1 import run
from ppo import parse_emergency
from record_dayoff_games import mark_off
from search_eval import Policy
from starts import capture


def outcome(fin, c, i):
    silver = [d for d, t in c.opens[i] if t >= 3]
    return {"final": int(fin[i]), "start": int(c.start[i]), "second": len(silver) >= 2,
            "second_day": silver[1] if len(silver) >= 2 else None, "toss": c.spend[i]["버리기"], "prep": c.spend[i]["상자 이동·합성"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("out")
    p.add_argument("--name-a", default="cn1 30M (학습 전)")
    p.add_argument("--name-b", default="cn3 10M (은상자 연습 10M)")
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--pool-ckpt", default="runs/cn1/agent_30M.pt")
    p.add_argument("--pool-seed", type=int, default=4321)
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    hold, k, _ = capture(Policy(args.pool_ckpt, dev), args.pool_games, args.pool_seed, args.pool, env_kw={"open_min_tier": 2})
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5")}
    pols = {"A": Policy(args.a, dev), "B": Policy(args.b, dev)}
    n = args.games
    res = {x: run(pols[x], n, args.seed, hold, k, 3, rule, args.pool_seed) for x in "AB"}
    oa = [outcome(res["A"][0], res["A"][2], i) for i in range(n)]
    ob = [outcome(res["B"][0], res["B"][2], i) for i in range(n)]
    life = lambda o: o["final"] - o["start"]

    picks = []
    def take(label, cand):
        for i in cand:
            if i not in [j for j, _ in picks]:
                picks.append((int(i), label))
                return

    idx = range(n)
    take("A는 두 번째 은상자까지, B는 스왑을 버리기에 씀",
         sorted([i for i in idx if oa[i]["second"] and not ob[i]["second"]], key=lambda i: -ob[i]["toss"]))
    take("A는 두 번째 은상자까지, B는 스왑을 버리기에 씀 (둘째 사례)",
         sorted([i for i in idx if oa[i]["second"] and not ob[i]["second"]], key=lambda i: -ob[i]["toss"]))
    take("A는 순환했고 B가 먼저 죽음",
         sorted([i for i in idx if oa[i]["second"] and life(ob[i]) < life(oa[i])], key=lambda i: life(ob[i]) - life(oa[i])))
    take("B가 훨씬 오래 삶 (A는 순환 없이 죽음)",
         sorted([i for i in idx if not oa[i]["second"]], key=lambda i: life(oa[i]) - life(ob[i])))
    take("둘 다 두 번째 은상자까지", [i for i in idx if oa[i]["second"] and ob[i]["second"]])

    sel = [i for i, _ in picks]
    frames = {}
    for x in "AB":
        fr = [[] if i in sel else None for i in range(n)]
        fin2, _, _ = run(pols[x], n, args.seed, hold, k, 3, rule, args.pool_seed, frames=fr)
        assert (fin2 == res[x][0]).all(), "다시 둔 결과가 처음과 다르다"
        frames[x] = fr
    games = []
    for i, label in picks:
        fa, fb = mark_off(frames["A"][i]), mark_off(frames["B"][i])
        div = next((t for t, (u, v) in enumerate(zip(fa, fb)) if u["a"] != v["a"]), 0)
        a, b = oa[i], ob[i]
        note = (f"{label} · {a['start']}일 시작 · A: {a['final']}일 사망, 둘째 은상자 {a['second_day'] or '없음'}, 첫 4일 버리기 {a['toss']}·상자 준비 {a['prep']}"
                f" / B: {b['final']}일 사망, 둘째 은상자 {b['second_day'] or '없음'}, 첫 4일 버리기 {b['toss']}·상자 준비 {b['prep']}")
        games.append({"id": str(i), "note": note, "search": {"day": a["final"], "frames": fa}, "base": {"day": b["final"], "frames": fb}, "diverge": div})
        print(f"게임 {i}: {note} · 첫 갈림 {div}번째 행동")
    ui = {"title": "은상자 연습 리플레이",
          "sub": "은상자 연습 1단계 평가 상태(학습에 쓰지 않은 풀, cn1 30M의 실제 게임 낮 상태에서 1~6행 자원 하나를 은상자로 바꿈)에서, "
                 f"같은 시작 상태·같은 난수로 두 모델이 둔 기록입니다. <b>A</b>: {args.name_a}. <b>B</b>: {args.name_b}. "
                 "규칙은 둘 다 은상자부터 개봉, 동상자는 비상 예외(스왑 0·하트 5 이하·열 수 있는 은·금 없음)만, 휴식일 버리기 금지입니다. "
                 "기록은 연습 시작 상태부터이고, '첫 갈림으로'를 누르면 두 모델이 처음 다른 수를 둔 곳으로 갑니다.",
          "a": f"A: {args.name_a}", "b": f"B: {args.name_b}", "aShort": "A", "bShort": "B"}
    with open(args.out, "w") as f:
        json.dump({"ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
