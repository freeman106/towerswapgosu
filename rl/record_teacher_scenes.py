"""교사 대표 장면 리플레이: 교사를 붙인 정책(cn7_nf 20M + 상자 합성 계획 + 1~2수 생산·합성 교사)으로 정상 시작을 두고,
교사가 기존 선택을 바꾼 결정 중 종류별 대표를 고른다. 장면마다 그 결정 상태에서
  기존 선택 줄(정책이 고른 수 뒤 정책) / 교사 줄(교사 행동열 뒤 정책)을
교사가 비교에 쓴 가상 난수 첫 묶음(같은 시드·같은 샘플링 잡음)으로 그날 밤이 끝날 때까지 진행한 프레임을 짝으로 저장한다
(build_replay.py 입력, search 칸 = 교사 줄, base 칸 = 기존 선택 줄). 설명에는 교사가 본 가상 8묶음 평균과 이 묶음의 결과를 적는다.
먼저 한 번 두어 결정을 고르고, 같은 조건으로 다시 두며 고른 결정에서만 두 줄을 기록한다(고른 결정이 처음과 같은지 확인).
사용: python rl/record_teacher_scenes.py out.json [--games 32 --seed 4242]"""
import argparse
import json

import numpy as np
import torch

from diag_bronze_flow import run
from eval_restrict import CONDS
from ppo import parse_emergency
from search_eval import Policy
from teacher import Teacher

WANT = (("한 수 생산", lambda d, c: c["종류"] == "한 수·생산"),
        ("한 수 기본→동 합성", lambda d, c: c["종류"] == "한 수·합성"),
        ("준비 1수 → 생산", lambda d, c: c["종류"] == "두 수·생산" and d.get("완성")),
        ("준비 1수 → 합성", lambda d, c: c["종류"] == "두 수·합성" and d.get("완성")),
        ("성 보수 대신 교사 수", lambda d, c: d["candidates"][0]["수"][0].endswith("성")),
        ("두 수 계획 재확인 실패", lambda d, c: c["스왑"] == 2 and d.get("완성") is False))


def chosen_of(d):
    ok = [c for c in d["candidates"][1:] if c["판정"] == "통과"]
    return max(ok, key=lambda c: c["ΔQ"])


def stat(c):
    b = c["배치(기본/동/은+/5~6행/얼음벽)"]
    return (f"사망 {c['사망']}/8 · 밤 직전 하트 {c['밤 직전 하트']} · 밤 피해 {c['밤 피해']} · 아침 하트 {c['아침 하트']} · "
            f"아침 공격 무기 기본 {b[0]}·동 {b[1]}·5~6행 {b[3]} · 얼음벽 {b[4]}")


def line_result(fr):
    dusk = next((f["h"] for f in fr if f["a"] == 224), None)
    last = fr[-1]
    dead = last["a"] == 224 and last["p"] == 25
    return dusk, (0 if dead else last["h"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ckpt", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--min-day", type=int, default=3)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    rule = {"open_min_tier": 3, "emergency": parse_emergency("2,0,5"), "merge_rule": (20, 6), **CONDS["B"][1]}
    n = args.games
    c1 = Teacher(n, pol, args.seed, True)
    run(pol, n, args.seed, rule, None, ctl=c1)
    rep = [d for d in c1.dec if d["chosen"]]
    picks = []
    for label, want in WANT:
        cand = [d for d in rep if d["day"] >= args.min_day and want(d, chosen_of(d)) and (d["game"], d["step"]) not in [(x["game"], x["step"]) for _, x in picks]]
        if not cand and label == "두 수 계획 재확인 실패":
            cand = [d for d in rep if want(d, chosen_of(d))]
        if cand:
            picks.append((label, max(cand, key=lambda d: chosen_of(d)["ΔQ"])))
    keys = [(d["game"], d["step"]) for _, d in picks]
    c2 = Teacher(n, pol, args.seed, True, scene_keys=keys)
    fin, _, _ = run(pol, n, args.seed, rule, None, ctl=c2)
    got = {(sc["game"], c2.dec[sc["dec"]]["step"]): sc for sc in c2.scenes}
    games, table = [], []
    for label, d in picks:
        sc = got[(d["game"], d["step"])]
        d2 = c2.dec[sc["dec"]]
        assert d2["chosen"] == d["chosen"], "다시 둔 결정이 처음과 다르다"
        base, ch = d["candidates"][0], chosen_of(d)
        fb, ft = sc["frames"]
        (db, mb), (dt, mt) = line_result(fb), line_result(ft)
        seq = " → ".join(ch["수"])
        note = (f"{d['day']}일, 하트 {d['hearts']} · 스왑 {d['swaps']} | 기존 선택: {base['수'][0]} | 교사 행동열({ch['종류']}, 스왑 {ch['스왑']}): {seq} "
                f"→ 결과 무기 {[f'{w[0]}{w[1]}({w[2]},{w[3]})' for w in ch['결과 무기']]}, 사라진 무기 {[f'{w[0]}{w[1]}' for w in ch['사라진 무기']] or '없음'}, "
                f"자원 칸 {ch['자원 칸 변화']:+.1f}"
                + (f", 둘째 수 실제 재확인 {'유효' if d.get('완성') else '실패'}" if ch["스왑"] == 2 else "")
                + f" | 가상 8묶음 — 기존: {stat(base)} / 교사: {stat(ch)}, ΔQ {ch['ΔQ']:+.3f} ± {ch['±']:.3f}"
                + f" | 이 화면(묶음 1): 밤 직전 하트 기존 {db} · 교사 {dt}, 다음 아침 하트 기존 {mb} · 교사 {mt}")
        nights = lambda fr: [[t, "밤 시작"] for t, f in enumerate(fr) if f["a"] == 224]
        mt_ = [[0, "교사: 첫 수"]] + ([[1, "교사: 둘째 수"]] if ch["스왑"] == 2 else []) + nights(ft)
        mb_ = [[0, "기존 선택"]] + nights(fb)
        games.append({"id": f"{d['game']}-{d['step']}", "label": f"{label} · {d['day']}일", "note": note, "diverge": 0,
                      "search": {"day": int(fin[d["game"]]), "frames": ft, "marks": mt_},
                      "base": {"day": int(fin[d["game"]]), "frames": fb, "marks": mb_}})
        table.append({"장면": label, "게임": d["game"], "행동": d["step"] + 1, "일": d["day"], "하트": d["hearts"], "스왑": d["swaps"],
                      "기존 선택": base["수"][0], "교사 행동열": seq, "종류": ch["종류"], "결과 무기": ch["결과 무기"], "사라진 무기": ch["사라진 무기"],
                      "자원 칸": ch["자원 칸 변화"], "완성": d.get("완성"), "가상 기존": base, "가상 교사": ch,
                      "묶음1": {"기존": [db, mb], "교사": [dt, mt]}})
        print(f"[{label}] 게임 {d['game']} 행동 {d['step'] + 1}: {note}\n")
    ui = {"title": "생산·합성 교사 장면", "flip": "교사가 바꾼 수",
          "sub": "기존 정책(cn7_nf 20M, 대포 방향 금지, 상자 합성 계획 켬)에 1~2수 생산·합성 교사를 붙여 둘 때, 교사가 기존 선택을 바꾼 결정의 장면입니다. "
                 "같은 결정 상태에서 <b>기존 선택</b>으로 이어 간 줄과 <b>교사 행동열</b>로 이어 간 줄을, 교사가 비교에 쓴 가상 난수 첫 묶음(같은 새 타일·같은 밤)으로 "
                 "그날 밤이 끝날 때까지 보여 줍니다. 두 줄 모두 첫 수(교사는 두 수) 뒤에는 같은 정책이 둡니다. "
                 "설명의 '가상 8묶음'은 교사가 결정할 때 본 평균입니다. 교사는 사망이 늘지 않고 다음 아침 하트가 줄지 않으며 가치(Q)가 유의하게 높을 때만 바꿉니다. "
                 f"정상 시작 시드 {args.seed}, {n}판에서 종류별로 골랐습니다.",
          "a": "교사 행동열", "b": "기존 선택", "aShort": "교사", "bShort": "기존"}
    with open(args.out, "w") as f:
        json.dump({"ui": ui, "games": games}, f, separators=(",", ":"), ensure_ascii=False)
    with open(args.out.replace(".json", "_table.json"), "w") as f:
        json.dump(table, f, ensure_ascii=False, default=int)


if __name__ == "__main__":
    main()
