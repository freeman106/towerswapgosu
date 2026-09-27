"""search_eval.py --log 기록 분석: 탐색이 어떤 상황에서 어떤 행동을 바꿨는지, 게임 흐름(하트·돌 넣기·타워 등급)이 어떻게 달라졌는지.
사용: python rl/analyze_search.py log.json"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np


def bucket(v, edges, names):
    for e, nm in zip(edges, names):
        if v < e:
            return nm
    return names[-1]


def main():
    L = json.load(open(sys.argv[1]))
    dec = L["search"]["dec"]
    n = len(dec)
    ch = [d for d in dec if d["changed"]]
    print(f"탐색한 결정 {n} · 바뀐 결정 {len(ch)} ({len(ch) / n:.1%})")

    print("\n[기본 행동 종류별] 결정 수 · 바뀐 비율 · 바뀌면 주로 무엇으로")
    by = defaultdict(list)
    for d in dec:
        by[d["base"]].append(d)
    for cat, ds in sorted(by.items(), key=lambda kv: -len(kv[1])):
        c = [d for d in ds if d["changed"]]
        to = Counter(d["chosen"] for d in c).most_common(3)
        print(f"  {cat:14s} {len(ds):6d} · 바뀜 {len(c) / len(ds):5.1%} · " + ", ".join(f"{k} {v / max(len(c), 1):.0%}" for k, v in to))

    print("\n[바뀐 방향] 기본 → 탐색 (상위 12) · 건수 · 평균 이득(Q) · 원래 확률 순위 평균")
    tr = defaultdict(list)
    for d in ch:
        tr[(d["base"], d["chosen"])].append(d)
    for (a, b), ds in sorted(tr.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {a:14s} → {b:14s} {len(ds):5d} · 이득 {np.mean([d['gain'] for d in ds]):.3f} · 순위 {np.mean([d['rank'] for d in ds]):.1f}")

    print("\n[상황별] 결정 수 · 바뀐 비율 · 바뀐 결정의 평균 이득")
    for name, key, edges, names in (
        ("일차", "day", (10, 20, 26, 30, 99), ("1~9", "10~19", "20~25", "26~29", "30+")),
        ("하트", "hearts", (6, 11, 21, 31, 999), ("1~5", "6~10", "11~20", "21~30", "31+")),
        ("타워 점유율", "occ", (.5, .65, .75, 9), ("<50%", "50~65%", "65~75%", "75%+")),
    ):
        g = defaultdict(list)
        for d in dec:
            g[bucket(d[key], edges, names)].append(d)
        print(f"  {name}: " + " | ".join(
            f"{nm} {len(g[nm])} · {np.mean([d['changed'] for d in g[nm]]):.0%} · {np.mean([d['gain'] for d in g[nm] if d['changed']] or [0]):.3f}"
            for nm in names if g[nm]))

    print("\n[게임 흐름] 5일 구간 평균, 기본 / 탐색 (그 구간까지 살아 있는 게임 기준)")
    for cond in ("base", "search"):
        dusk, morn, fin = L[cond]["dusk"], L[cond]["morning"], L["final"][cond]
        mh = {(m["game"], m["day"]): m["hearts"] for m in morn}
        for d in dusk:
            nxt = mh.get((d["game"], d["day"] + 1))
            d["loss"] = d["hearts"] if fin[d["game"]] == d["day"] else (d["hearts"] - nxt if nxt is not None else np.nan)
    print("  구간     | 밤 직전 하트 | 하루 돌 넣기 | 밤 손실 | 타워 점유율 | 공격 타워 기본/동/은")
    for lo in range(1, 36, 5):
        row = []
        for cond in ("base", "search"):
            ds = [d for d in L[cond]["dusk"] if lo <= d["day"] < lo + 5]
            row.append(ds)
        if not row[0] or not row[1]:
            continue
        f = lambda ds, k: np.nanmean([d[k] for d in ds])
        t = lambda ds: "/".join(f"{np.mean([d['tiers'][x] for d in ds]):.1f}" for x in (1, 2, 3))
        print(f"  {lo:2d}~{lo + 4:2d}일 | {f(row[0], 'hearts'):5.1f} / {f(row[1], 'hearts'):5.1f} | {f(row[0], 'stones'):4.1f} / {f(row[1], 'stones'):4.1f} | "
              f"{f(row[0], 'loss'):4.2f} / {f(row[1], 'loss'):4.2f} | {f(row[0], 'occ'):4.0%} / {f(row[1], 'occ'):4.0%} | {t(row[0])} / {t(row[1])}")


if __name__ == "__main__":
    main()
