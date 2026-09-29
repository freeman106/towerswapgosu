"""금지 조건 적응 학습 리플레이용 짝 기록: 같은 정상 시작 게임(같은 게임 시드·게임별 샘플링 잡음)을
A = cn7_nf 20M(대포 방향 금지) / B = cn7_nft 20M(대포 방향·모든 버리기 금지)이 각자 학습한 제한과
상자 합성 계획(연쇄 포함 한 수 합성 + 준비 이동 1수)으로 둔 기록을 저장한다(build_replay.py 입력, search 칸 = A, base 칸 = B).
먼저 프레임 없이 두어 사례를 고르고, 고른 게임만 프레임을 기록해 다시 둔다(결과가 처음과 같은지 확인).
사용: python rl/record_restrict_games.py out.json [--seed 12345]"""
import argparse
import json

import numpy as np
import torch

from diag_bronze_flow import run
from eval_restrict import CONDS, Restrict
from ppo import parse_emergency
from record_dayoff_games import mark_off
from search_eval import Policy

MODELS = {"A": ("runs/cn7_nf/agent_20M.pt", "B", "A: 대포 방향 금지 학습 +10M"),
          "B": ("runs/cn7_nft/agent_20M.pt", "D", "B: 대포 방향·버리기 금지 학습 +10M")}


def summary(fin, fins, c, i):
    """게임 i의 요약: 최종 일차, 게임-날당 매치 없는 이동·버리기·성 보수, 무기 생산·합성, 첫 은상자, 비상 개봉"""
    days = max(1, fin[i] - c.start[i] + 1)
    d = c.drags[i]
    made = np.array(fins[i]["made_tiers"])
    return {"final": int(fin[i]), "noop": d["매치 없는 이동"] / days, "toss": d["버리기"] / days, "repair": d["성 보수"] / days,
            "prod": made[:3, 0].sum() / days, "merge": made[:3, 1:].sum() / days, "wall": int(made[3].sum()),
            "silver": int(fins[i]["opened"][2] > 0), "emerg": int(fins[i]["emergency_opens"])}


def marks_of(c, i, n_frames):
    g, m = c.g[i], []
    for l in g["losses"]:
        m.append((l["step"], f"동상자 잃음: {l['cause']} ({l['b']}→{l['b_after']}개, 하트 {l['hearts']})"))
    if g["held2"] is not None:
        m.append((g["held2"]["step"], "동상자 2개 동시 보유"))
    if g["held3"] is not None:
        m.append((g["held3"]["step"], "동상자 3개 동시 보유"))
    for st, _, kind, _, _ in c.log[i]:
        m.append((st, "합성 계획: " + ("은상자 합성 수" if kind == "합성" else "준비 이동")))
    if g["silver"] is not None:
        m.append((g["silver"]["step"], f"은상자 합성({g['silver']['type']})"))
    if g["opened"] is not None:
        m.append((g["opened"]["step"], "합성 은상자 개봉"))
    m.append((n_frames - 1, "마지막 행동"))
    out, seen = [], set()
    for st, t in sorted(m):
        if (st, t) not in seen:
            seen.add((st, t))
            out.append([int(st), t])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    base = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6)}
    n = args.games
    pols = {k: Policy(v[0], dev) for k, v in MODELS.items()}
    rule = {k: {**base, **CONDS[v[1]][1]} for k, v in MODELS.items()}
    res = {}
    for k in MODELS:
        c = Restrict(n)
        fin, fins, _ = run(pols[k], n, args.seed, rule[k], None, ctl=c)
        c.finish()
        res[k] = (fin, fins, c)
    S = {k: [summary(*res[k], i) for i in range(n)] for k in MODELS}
    fa, fb = res["A"][0], res["B"][0]
    ma, mb = np.median(fa), np.median(fb)
    print(f"A 중앙 {ma:.0f}일, B 중앙 {mb:.0f}일")

    picks = []

    def take(label, cand):
        for i in cand:
            if i not in [j for j, _ in picks]:
                picks.append((int(i), label))
                return

    idx = np.arange(n)
    take("보통 판: 두 모델 모두 중앙값 근처", sorted(idx, key=lambda i: abs(fa[i] - ma) + abs(fb[i] - mb)))
    take("B 조기 사망: A는 중앙값 이상, B는 매치 없는 이동이 많음",
         sorted([i for i in idx if fb[i] <= 7 and fa[i] >= ma], key=lambda i: -S["B"][i]["noop"]))
    take("A는 은상자까지, B는 못 감",
         sorted([i for i in idx if S["A"][i]["silver"] and not S["B"][i]["silver"]], key=lambda i: abs(fb[i] - mb)))
    take("B가 A보다 오래 삶 (A도 10일 이상)", sorted([i for i in idx if fa[i] >= 10], key=lambda i: fa[i] - fb[i]))
    take("B 최장 생존", sorted(idx, key=lambda i: -fb[i]))

    sel = [i for i, _ in picks]
    frames = {}
    for k in MODELS:
        fr = [[] if i in sel else None for i in range(n)]
        c = Restrict(n)
        fin2, _, _ = run(pols[k], n, args.seed, rule[k], None, frames=fr, ctl=c)
        assert (fin2 == res[k][0]).all(), "다시 둔 결과가 처음과 다르다"
        for i in sel:
            for st, a0, _ in c.g[i]["forced"]:
                fr[i][st]["base"] = a0
            mark_off(fr[i])
        frames[k] = (fr, c)
    games = []
    fmt = lambda s: (f"{s['final']}일 사망 · 매치 없는 이동 {s['noop']:.1f}/일 · 버리기 {s['toss']:.1f}/일 · 성 보수 {s['repair']:.1f}/일 · "
                     f"무기 생산 {s['prod']:.1f}·합성 {s['merge']:.2f}/일 · 얼음벽 {s['wall']}개 · 비상 개봉 {s['emerg']} · 은상자 {'O' if s['silver'] else 'X'}")
    for i, label in picks:
        fa_, ca = frames["A"][0][i], frames["A"][1]
        fb_, cb = frames["B"][0][i], frames["B"][1]
        note = f"{label} · A: {fmt(S['A'][i])} / B: {fmt(S['B'][i])}"
        div = next((t for t, (u, v) in enumerate(zip(fa_, fb_)) if u["a"] != v["a"]), 0)
        games.append({"id": str(i), "label": label, "note": note, "diverge": div,
                      "search": {"day": int(fa[i]), "frames": fa_, "marks": marks_of(ca, i, len(fa_))},
                      "base": {"day": int(fb[i]), "frames": fb_, "marks": marks_of(cb, i, len(fb_))}})
        print(f"게임 {i}: {note} · 첫 갈림 {div + 1}번째 행동")
    ui = {"title": "버리기 금지 적응 리플레이", "flip": "합성 계획이 바꾼 수",
          "sub": "cn7 10M에서 같은 설정으로 10M 더 학습한 두 분기가 같은 정상 시작 게임(같은 시드·같은 샘플링 잡음)을 둔 기록입니다. "
                 "<b>A</b>: 대포 방향 전환만 금지하고 학습. <b>B</b>: 대포 방향 전환과 모든 버리기를 금지하고 학습. "
                 "평가도 각자 학습한 제한 그대로이고, 두 모델 모두 상자 합성 계획(연쇄 포함 한 수 은상자 합성 + 준비 이동 1수)을 켰습니다. "
                 f"상자 개봉은 은상자부터(동상자는 비상 예외만)입니다. 시드 {args.seed} 1,024판 중 5판을 골랐습니다.",
          "a": MODELS["A"][2], "b": MODELS["B"][2], "aShort": "A", "bShort": "B"}
    with open(args.out, "w") as f:
        json.dump({"ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
