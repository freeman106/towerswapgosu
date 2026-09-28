"""연습 시작 상태 모으기: 정책이 실제로 둔 게임에서 낮 상태(입력 대기, 스왑 남음, 휴식일 아님)를 골라 SearchEnv 슬롯에 담는다.
일차 구간마다 같은 수를 목표로 삼아(늦은 구간은 살아 있는 게임이 적어 덜 찰 수 있다) 초반 상태에만 몰리지 않게 한다.
담은 상태는 VecEnv.add_starts(hold, 슬롯, 난이도, place_tier)로 연습 시작 풀에 넣는다(place_tier=3이면 자원 하나를 은상자로)."""
import numpy as np

import towerswap as ts
from search_eval import play

BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 99))


class Capture:
    """play()의 ctl: 입력 대기 낮 상태를 확률 p로 골라 일차 구간별 목표 수까지 담는다"""

    def __init__(self, hold, count, p, seed):
        self.hold, self.p, self.seed = hold, p, seed
        self.quota = {b: count // len(BUCKETS) for b in BUCKETS}
        self.filled = 0
        self.info = []  # (게임, 일차, 하트)
        self.step = 0

    def before(self, env, alive):
        u = np.random.default_rng([self.seed, self.step, 3]).random(len(alive))
        for i in np.nonzero(alive & (u < self.p))[0]:
            day, hearts, swaps, phase, _ = env.state(int(i))
            b = next(b for b in BUCKETS if b[0] <= day <= b[1])
            if self.quota[b] <= 0 or phase != 0 or swaps <= 0 or (day % 10 == 1 and env.made(int(i))[2] == day):
                continue
            if not any(c[:1] in "lsigd" for row in env.board(int(i))[0][:6] for c in row):
                continue  # 은상자로 바꿀 자원이 없는 상태는 담지 않는다 (있음/없음 짝이 어긋나지 않게)
            seed = np.random.default_rng([self.seed, self.filled, 5]).integers(1, 2 ** 62, size=1, dtype=np.uint64)
            self.hold.store(env, np.array([i], np.int64), np.array([self.filled], np.int64), seed)
            self.info.append((int(i), day, hearts))
            self.quota[b] -= 1
            self.filled += 1
        self.step += 1
        return {}, np.zeros(len(alive), bool)

    def after(self, env, i, a):
        pass


def capture(pol, n, seed, count, p=0.03, env_kw=None):
    """정책 pol로 n판을 두며 상태를 최대 count개 담는다. (SearchEnv, 담은 수, [(게임, 일차, 하트)])"""
    hold = ts.SearchEnv(count)
    cap = Capture(hold, count, p, seed)
    play(pol, n, seed, False, 0, 0, 0, None, ctl=cap, no_toss=True, env_kw=env_kw)
    return hold, cap.filled, cap.info


def describe(info):
    days = np.array([d for _, d, _ in info])
    hearts = np.array([h for _, _, h in info])
    parts = " · ".join(f"{lo}~{hi if hi < 99 else ''}일 {np.sum((days >= lo) & (days <= hi))}" for lo, hi in BUCKETS)
    return f"상태 {len(info)}개 (게임 {len({g for g, _, _ in info})}판) · 일차 {parts} · 하트 평균 {hearts.mean():.1f} (1~5: {np.mean(hearts <= 5):.0%})"
