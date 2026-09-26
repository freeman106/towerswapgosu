"""배치 환경 처리량 측정: 무작위 유효 행동으로 초당 스텝 수를 잰다.
사용: python rl/bench_env.py [환경 수] [스레드 수] [초]"""
import sys
import time

import numpy as np
import towerswap as ts

n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

env = ts.VecEnv(n, seed=1, threads=threads)
grid = np.zeros((n, *ts.GRID_SHAPE), np.float32)
scal = np.zeros((n, ts.N_SCALAR), np.float32)
mask = np.zeros((n, ts.N_ACTIONS), bool)
rew = np.zeros(n, np.float32)
done = np.zeros(n, bool)
days = np.zeros(n, np.float32)
env.reset(grid, scal, mask)
rng = np.random.default_rng(0)

steps, t_env, t0 = 0, 0.0, time.perf_counter()
while time.perf_counter() - t0 < secs:
    act = np.argmax(rng.random(mask.shape, dtype=np.float32) * mask, axis=1)
    t = time.perf_counter()
    env.step(act, grid, scal, mask, rew, done, days)
    t_env += time.perf_counter() - t
    steps += n
el = time.perf_counter() - t0
fin = env.pop_finished()
days_avg = np.mean([f["day"] for f in fin]) if fin else 0
print(f"envs={n} threads={threads or 'all'}: {steps / el:,.0f} steps/s (env만 {steps / t_env:,.0f}/s), "
      f"끝난 게임 {len(fin)}개, 평균 도달 day {days_avg:.1f}")
