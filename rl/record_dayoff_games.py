"""휴식일 리플레이용 짝 기록: 31일 휴식일 제안 상태(dayoff.py와 같은 되돌린 상태·가상 난수)에서
A = 새 규칙(수락, 버리기 금지, 그날은 정책만) / B = 지금 방식(수락 후 탐색, 버리기 허용)으로 이어 둔 기록을
원래 게임의 제안 전 기록 뒤에 붙여 저장한다(build_replay.py 입력 형식, search 칸 = A, base 칸 = B).
사용: python rl/record_dayoff_games.py rl/runs/long1/agent_40M.pt rl/runs/dayoff_frames4242.pkl out.json [--check a.json:mode,b.json:mode]"""
import argparse
import json
import pickle

import numpy as np
import torch

import towerswap as ts
from dayoff import OFFER, SEARCH, YES, DayOffControl, towers
from rescue import restore
from search_eval import Policy, play

CONDS = (("A", "policy", True), ("B", "now", False))  # (이름, dayoff.py 조건, 버리기 금지)


def mark_off(frames):
    """휴식일(수락한 날)의 프레임에 off=1을 붙인다"""
    off_day = None
    for f in frames:
        if off_day is not None and f["d"] == off_day and f["p"] != OFFER:
            f["off"] = 1
        if f["p"] == OFFER and f["a"] == YES:
            off_day = f["d"]
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("frames")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--copies", type=int, default=8)
    p.add_argument("--day", type=int, default=31)
    p.add_argument("--pick", type=int, default=5)
    p.add_argument("--check", default="runs/dayoff2.json:policy,runs/dayoff1.json:now", help="A, B 결과가 이 기록과 같은지 확인")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    with open(args.frames, "rb") as f:
        frames, final = pickle.load(f)
    M = args.copies
    points = [(i, t) for i, fr in enumerate(frames) for t, f in enumerate(fr) if f["p"] == OFFER and f["d"] == args.day]
    P = len(points)
    hold = restore(frames, args.seed, points, M)
    fin, fr, ctls = {}, {}, {}
    for name, mode, no_toss in CONDS:
        e2 = ts.VecEnv(P * M, seed=1, no_toss_day_off=no_toss)
        e2.load_from(hold, np.arange(P * M, dtype=np.int64), np.arange(P * M, dtype=np.int64))
        fr[name] = [[] for _ in range(P * M)]
        ctls[name] = DayOffControl(mode, P * M)
        st = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
        fin[name] = play(pol, P * M, args.seed + 7, True, *SEARCH, st, 2.0, env=e2, ctl=ctls[name], frames=fr[name])
        print(f"{name} ({mode}): 그 밤 생존 {np.mean(fin[name] > args.day):.1%} · 평균 최종 {fin[name].mean():.2f}일", flush=True)
    if args.check:
        for (name, _, _), spec in zip(CONDS, args.check.split(",")):
            path, mode = spec.split(":")
            ref = json.load(open(path))["res"][mode]["final"]
            assert fin[name].tolist() == ref, f"{name}: {path}의 {mode} 결과와 다르다"
        print("재현 확인: 이전 실험 결과와 같음", flush=True)

    # 고르기 (제안 상태가 겹치지 않게)
    a, b = fin["A"], fin["B"]
    D = args.day
    tb = np.array([towers(frames[points[j // M][0]][points[j // M][1]]["b"]) - (ctls["B"].dusk[j] or (0,))[0] for j in range(P * M)])
    picks, used = [], set()

    def take(label, cand):
        for j in cand:
            if j // M not in used:
                used.add(j // M)
                picks.append((int(j), label))
                return

    J = np.arange(P * M)
    take("지금 방식이 버리기를 많이 하고 31일 사망, 새 규칙은 생존", sorted(J[(b == D) & (a > D)], key=lambda j: -ctls["B"].toss[j]))
    take("지금 방식이 방어물을 많이 잃고 31일 사망, 새 규칙은 생존", sorted(J[(b == D) & (a > D)], key=lambda j: -tb[j]))
    take("둘 다 31일 사망 (새 규칙도 휴식일에 수를 많이 둠)", sorted(J[(a == D) & (b == D)], key=lambda j: -ctls["A"].acts[j]))
    take("새 규칙이 31일 사망, 지금 방식은 생존", sorted(J[(a == D) & (b > D)], key=lambda j: -b[j]))
    take("새 규칙이 가장 오래 산 판", sorted(J, key=lambda j: -a[j]))
    for j in sorted(J[(b == D) & (a > D)], key=lambda j: -(a[j] - b[j])):
        if len(picks) >= args.pick:
            break
        take("지금 방식 31일 사망, 새 규칙 생존", [j])
    games = []
    for j, label in picks[:args.pick]:
        i, t = points[j // M]
        pre = mark_off([dict(f) for f in frames[i][:t]])
        fa, fb = pre + mark_off(fr["A"][j]), pre + mark_off(fr["B"][j])
        div = next((k for k, (x, y) in enumerate(zip(fa, fb)) if x["a"] != y["a"]), None)
        games.append({"id": f"{i}-{j % M}", "note": label, "search": {"day": int(a[j]), "frames": fa}, "base": {"day": int(b[j]), "frames": fb},
                      "diverge": div if div is not None else t})
        print(f"게임 {i} 복제 {j % M}: {label} · B {b[j]}일 → A {a[j]}일 · 휴식일 행동 A {ctls['A'].acts[j]} / B {ctls['B'].acts[j]}, "
              f"B 버리기 {ctls['B'].toss[j]} · 제안 {t}번째 행동, 첫 갈림 {div}")
    ui = {"title": "휴식일 리플레이",
          "sub": f"long1 40M 탐색 에이전트가 {D}일 휴식일 제안을 받은 상태(시드 {args.seed})에서, 같은 가상 난수로 두 방식을 이어 둔 기록입니다. "
                 "<b>새 규칙</b>: 수락, 휴식일 버리기 금지, 그날은 정책만(탐색 끔). <b>지금 방식</b>: 수락 후 휴식일에도 탐색, 버리기 허용. "
                 "휴식일 이후는 둘 다 같은 탐색 에이전트입니다. 제안 전까지는 원래 게임 기록이고, 휴식일 프레임은 단계 옆에 '휴식일'로 표시합니다. "
                 "'첫 갈림으로'를 누르면 두 기록이 처음 다른 수로 갑니다.",
          "a": "새 규칙 (버리기 금지·휴식일 정책만)", "b": "지금 방식 (휴식일에도 탐색)", "aShort": "새 규칙", "bShort": "지금 방식"}
    with open(args.out, "w") as f:
        json.dump({"model": args.ckpt, "ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
