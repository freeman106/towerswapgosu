"""은상자 개봉일 진단 (1단계 평가 풀, 같은 상태·같은 난수로 모델별 비교). 스왑은 다음 날로 넘어가지 않아 개봉일에 모두 써야 한다.
결정마다 유효한 드래그를 복제본에 실제로 적용해 결과를 엔진 통계로 판정한다(VecEnv.drag_outcomes):
  생산 = 기본 무기·얼음벽 생성, 동 합성 = 동 무기 생성, 은 합성 = 은 이상 무기 생성, 상자 합성, 성 보수, 버리기, 그 밖의 이동
1. 개봉 직후·개봉일 동안의 자원 칸 수와 생산 가능한 수(매치를 만드는 드래그 수), 생산 가능한 수가 없는 결정 비율
2. 합성 기회(동 합성·은 합성 가능한 드래그가 있음)에서 실제로 고른 것, 버리기를 고른 결정에 있던 기회
3. 개봉일에 무기를 합성한 수로 나눈 날들의 생산·상자·버리기·자원 칸·그날 밤 피해·이후 생존 (관찰 비교), 합성 직후 생산 가능한 수의 변화
사용: python rl/diag_open_day.py A.pt [B.pt ...] [--games 512]"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch

from eval_stage1 import run
from ppo import parse_emergency
from search_eval import Policy
from starts import capture

CATS = ("은 합성", "동 합성", "상자 합성", "생산", "성 보수", "버리기", "그 밖의 이동")


def category(row):
    if row[2] > 0 or row[3] > 0:
        return "은 합성"
    if row[1] > 0:
        return "동 합성"
    if row[5] > 0:
        return "상자 합성"
    if row[0] > 0 or row[4] > 0:
        return "생산"
    if row[7] > 0:
        return "성 보수"
    if row[6] > 0:
        return "버리기"
    return "그 밖의 이동"


class OpenDay:
    """play()의 ctl: 첫 은상자 개봉 뒤 그날의 입력 대기 결정마다 선택지 결과를 판정해 기록한다"""

    def __init__(self, n):
        self.open_day = np.full(n, -1)
        self.dec = []                       # (게임, 자원 칸, 생산 수, 동 합성 수, 은 합성 수, 상자 합성 수, 버리기 수, 고른 분류)
        self.per_game = defaultdict(Counter)  # 게임 → 개봉일 고른 분류 수
        self.res_after = {}                 # 게임 → 개봉 직후 자원 칸
        self.dusk_h = {}                    # 게임 → 개봉일 밤 직전 하트
        self.next_h = {}                    # 게임 → 다음 날 아침 하트
        self.seq = defaultdict(list)        # 게임 → 개봉일 결정 순서대로 (생산 수, 고른 분류)
        self.out = np.zeros((168, 8), np.int32)

    def before(self, env, alive):
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        i = int(i)
        day, hearts, _, phase, _ = env.state(i)
        od = self.open_day[i]
        if od >= 0 and day == od + 1 and i not in self.next_h:
            self.next_h[i] = hearts
        if phase not in (0, 25):
            return
        cells = env.board(i)[0]
        if od < 0:
            if 168 <= a < 216:
                y, x = (a - 168) // 6, (a - 168) % 6 + 1
                if y >= 1 and cells[y - 1][x - 1][:1] == "h" and int(cells[y - 1][x - 1][1]) >= 3:
                    self.open_day[i] = day
            return
        if day != od:
            return
        if a == 224 or phase == 25:
            self.dusk_h[i] = hearts
        if phase != 0:
            return
        res = sum(1 for row in cells[:6] for c in row if c[:1] in "lsigd")
        self.res_after.setdefault(i, res)
        env.drag_outcomes(i, self.out)
        v = self.out[self.out[:, 0] >= 0]
        n_prod = int(((v[:, 0] > 0) | (v[:, 4] > 0)).sum())
        n_b, n_s = int((v[:, 1] > 0).sum()), int(((v[:, 2] > 0) | (v[:, 3] > 0)).sum())
        n_c, n_t = int((v[:, 5] > 0).sum()), int((v[:, 6] > 0).sum())
        cat = category(self.out[a]) if a < 168 else "탭·기타"
        self.dec.append((i, res, n_prod, n_b, n_s, n_c, n_t, cat))
        self.per_game[i][cat] += 1
        self.seq[i].append((n_prod, cat, res))


def report(name, fin, fins, c, n):
    op = np.nonzero(c.open_day >= 0)[0]
    d = c.dec
    print(f"\n=== {name}: 은상자 연 판 {len(op)}/{n}, 개봉일 입력 대기 결정 {len(d)}개 (판당 {len(d) / max(len(op), 1):.0f}) ===")
    res0 = np.array([c.res_after[i] for i in op if i in c.res_after])
    arr = np.array([x[1:7] for x in d], float)
    print(f"  1. 자원 칸(1~6행): 개봉 직후 평균 {res0.mean():.1f} (중앙 {np.median(res0):.0f}, 6개 이하 {np.mean(res0 <= 6):.0%}) · 개봉일 결정 평균 {arr[:, 0].mean():.1f}")
    print(f"     생산 가능한 수(매치로 기본 무기·얼음벽이 생기는 드래그): 평균 {arr[:, 1].mean():.1f} · 0개인 결정 {np.mean(arr[:, 1] == 0):.0%} · "
          f"동 합성 가능 {arr[:, 2].mean():.1f}개 (있는 결정 {np.mean(arr[:, 2] > 0):.0%}) · 은 합성 가능 {arr[:, 3].mean():.1f}개 (있는 결정 {np.mean(arr[:, 3] > 0):.0%}) · "
          f"상자 합성 가능 {arr[:, 4].mean():.1f}개")
    for lo, hi in ((0, 6), (7, 10), (11, 99)):
        sel = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        if sel.any():
            cc = Counter(x[7] for x, s in zip(d, sel) if s)
            tot = sum(cc.values())
            print(f"     자원 칸 {lo}~{hi if hi < 99 else ''}개인 결정 {sel.mean():.0%}: 생산 가능 {arr[sel, 1].mean():.1f}개 · 고른 것 "
                  + ", ".join(f"{k} {cc[k] / tot:.0%}" for k in CATS if cc[k]))
    cc = Counter(x[7] for x in d)
    tot = sum(cc.values())
    print("  고른 것 (개봉일 전체): " + ", ".join(f"{k} {cc[k] / tot:.0%}" for k in CATS + ("탭·기타",) if cc[k]))
    for label, col in (("동 합성 기회가 있는 결정", 2), ("은 합성 기회가 있는 결정", 3)):
        sel = arr[:, col] > 0
        if sel.any():
            cs = Counter(x[7] for x, s in zip(d, sel) if s)
            t = sum(cs.values())
            print(f"  2. {label} {sel.sum()}개: " + ", ".join(f"{k} {cs[k] / t:.0%}" for k in CATS + ("탭·기타",) if cs[k]))
    toss = np.array([x[7] == "버리기" for x in d])
    if toss.any():
        print(f"     버리기를 고른 결정 {toss.sum()}개 중: 동 합성 기회 있음 {np.mean(arr[toss, 2] > 0):.0%} · 은 합성 기회 있음 {np.mean(arr[toss, 3] > 0):.0%} · "
              f"생산 가능한 수 있음 {np.mean(arr[toss, 1] > 0):.0%} · 셋 다 없음 {np.mean((arr[toss, 1] == 0) & (arr[toss, 2] == 0) & (arr[toss, 3] == 0)):.0%}")
    # 3. 개봉일 무기 합성 수로 나눈 비교
    groups = {"합성 0회": [], "합성 1~2회": [], "합성 3회+": []}
    for i in op:
        m = c.per_game[i]["동 합성"] + c.per_game[i]["은 합성"]
        groups["합성 0회" if m == 0 else "합성 1~2회" if m <= 2 else "합성 3회+"].append(i)
    print("  3. 개봉일 무기 합성 수별 (관찰 비교, 인과 아님):")
    for g, ids in groups.items():
        if not ids:
            continue
        pg = lambda k: np.mean([c.per_game[i][k] for i in ids])
        res = np.mean([np.mean([x[1] for x in d if x[0] == i]) for i in ids if any(x[0] == i for x in d)])
        loss = [c.dusk_h[i] - c.next_h.get(i, 0) for i in ids if i in c.dusk_h]
        life = np.mean([fin[i] - c.open_day[i] for i in ids])
        chest = np.mean([fins[i]["made"][1] for i in ids])
        print(f"     {g} ({len(ids)}판, {len(ids) / len(op):.0%}): 생산 {pg('생산'):.1f} · 동 합성 {pg('동 합성'):.1f} · 은 합성 {pg('은 합성'):.1f} · 상자 합성 {pg('상자 합성'):.1f} · "
              f"성 보수 {pg('성 보수'):.1f} · 버리기 {pg('버리기'):.1f} · 자원 칸 평균 {res:.1f} · 그날 밤 피해 {np.mean(loss):.2f} · 개봉 뒤 생존 {life:.1f}일 · 동상자 생성(게임 전체) {chest:.2f}")
    # 고른 수 직후 생산 가능한 수·자원 칸의 변화 (합성 대 다른 선택, 같은 판 개봉일 안에서)
    chg = defaultdict(lambda: ([], [], [], []))
    for i in op:
        s = c.seq[i]
        for t, (_, cat, _) in enumerate(s):
            if t + 3 < len(s):
                key = "무기 합성" if cat in ("동 합성", "은 합성") else cat
                b = chg[key]
                b[0].append(s[t][0])
                b[1].append(np.mean([s[t + k][0] for k in (1, 2, 3)]))
                b[2].append(s[t][2])
                b[3].append(np.mean([s[t + k][2] for k in (1, 2, 3)]))
    print("     고른 수 직후 3결정 평균 변화 (생산 가능한 수 · 자원 칸):")
    for key in ("무기 합성", "생산", "버리기", "성 보수", "그 밖의 이동"):
        if key in chg:
            b = chg[key]
            print(f"       {key} {len(b[0])}번: 생산 가능 {np.mean(b[0]):.1f} → {np.mean(b[1]):.1f} ({np.mean(b[1]) - np.mean(b[0]):+.2f}) · "
                  f"자원 칸 {np.mean(b[2]):.1f} → {np.mean(b[3]):.1f} ({np.mean(b[3]) - np.mean(b[2]):+.2f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
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
    for ck in args.ckpts:
        c = OpenDay(args.games)
        fin, fins, _ = run(Policy(ck, dev), args.games, args.seed, hold, k, 3, rule, args.pool_seed, ctl=c)
        report(ck, fin, fins, c, args.games)


if __name__ == "__main__":
    main()
