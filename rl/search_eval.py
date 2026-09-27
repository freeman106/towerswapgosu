"""추론 시 탐색 평가: 결정마다 후보 행동(기본 정책이 뽑은 행동 + 확률 상위 K개 + 그 밖의 유효 행동 R개)을 적용한 게임을
가상 난수 S묶음으로 복제하고, 같은 정책으로 그날 밤이 끝날 때까지 굴려 비교한다.
  Q = (밤까지 받은 보상 / σ) + γ^경과일 · V(끝 상태)   (사망하면 V 항은 0, σ는 체크포인트의 보상 정규화 표준편차)
후보들은 같은 가상 난수 묶음과 같은 정책 샘플링 잡음(검벨)을 쓴다. 본 게임도 환경마다 한 판씩, 판·스텝별 잡음을 고정해
두므로 기본 조건과 탐색 조건은 첫 탐색 개입 전까지 똑같다(짝 비교).
사용: python rl/search_eval.py rl/runs/cb9/agent.pt [--games 32] [--k 3] [--r 2] [--s 4]"""
import argparse
import time

import numpy as np
import torch

import towerswap as ts
from ppo import A, C, H, S, W, Agent

T_MAX = 400  # 굴리기 최대 행동 수 (넘으면 그 상태의 가치로 끊는다)


class Policy:
    def __init__(self, ckpt, dev):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        a = ck["args"]
        self.agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
        self.agent.load_state_dict(ck["agent"])
        self.agent.eval()
        self.dev = dev
        self.gamma = a["gamma"]
        self.sigma = float(np.sqrt(ck["rstd"][1] + 1e-8))

    @torch.no_grad()
    def __call__(self, grid, scal, mask, chunk=8192):
        """가려진 로짓과 가치 (numpy)"""
        lo, va = [], []
        for j in range(0, len(grid), chunk):
            l, v = self.agent(torch.from_numpy(grid[j:j + chunk]).to(self.dev), torch.from_numpy(scal[j:j + chunk]).to(self.dev))
            lo.append(l.masked_fill(~torch.from_numpy(mask[j:j + chunk]).to(self.dev), -1e9).cpu().numpy())
            va.append(v.reshape(-1).cpu().numpy())
        return np.concatenate(lo), np.concatenate(va)


def play(pol, n, seed, search, k, r, s, stats, z=0.0):
    env = ts.VecEnv(n, seed=seed)
    grid = np.zeros((n, C, H, W), np.float32)
    scal = np.zeros((n, S), np.float32)
    mask = np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    env.reset(grid, scal, mask)
    if search:
        m = n * (1 + k + r) * s
        se = ts.SearchEnv(m)
        sg = np.zeros((m, C, H, W), np.float32)
        ss = np.zeros((m, S), np.float32)
        sm = np.zeros((m, A), bool)
        sa = np.zeros(m, bool)
        ret, sdays, dead, lost = np.zeros(m, np.float32), np.zeros(m, np.float32), np.zeros(m, bool), np.zeros(m, np.float32)
    alive = np.ones(n, bool)
    final = np.zeros(n, int)
    step = 0
    while alive.any():
        logits, _ = pol(grid, scal, mask)
        act = (logits + np.random.default_rng([seed, step]).gumbel(size=(n, A))).argmax(1)
        if search:
            # 후보: 기본 행동, 확률 상위 k개, 그 밖의 유효 행동 r개
            rng = np.random.default_rng([seed, step, 11])
            idx, first, which = [], [], []
            for i in np.nonzero(alive & (mask.sum(1) >= 2))[0]:
                valid = np.nonzero(mask[i])[0]
                top = valid[np.argsort(-logits[i, valid])[:k]]
                cand = [int(act[i])] + [int(a) for a in top if a != act[i]]
                rest = [a for a in valid if a not in cand]
                cand += [int(a) for a in rng.choice(rest, size=min(r, len(rest)), replace=False)]
                for c in cand:
                    for j in range(s):
                        idx.append(i)
                        first.append(c)
                        which.append(j)
            if idx:
                idx, first, which = np.array(idx), np.array(first), np.array(which)
                seeds = np.random.default_rng([seed, step, 7]).integers(1, 2 ** 62, size=(n, s), dtype=np.uint64)
                u = len(idx)
                se.fork(env, idx.astype(np.int64), first.astype(np.int64), seeds[idx, which], sg, ss, sm, sa)
                noise = np.random.default_rng([seed, step, 9]).gumbel(size=(s, T_MAX, A)).astype(np.float32)
                t = 0
                while sa[:u].any() and t < T_MAX:
                    on = np.nonzero(sa[:u])[0]
                    lo, _ = pol(sg[on], ss[on], sm[on])
                    a_s = np.zeros(len(sa), np.int64)
                    a_s[on] = (lo + noise[which[on], t]).argmax(1)
                    se.step(a_s, sg, ss, sm, sa)
                    t += 1
                stats["rollout_steps"] += t
                se.results(ret, sdays, dead, lost)
                _, v_end = pol(sg[:u], ss[:u], sm[:u])
                q = ret[:u] / pol.sigma + pol.gamma ** sdays[:u] * v_end * (~dead[:u])
                # 게임별로 후보 × 난수 묶음 Q 표를 만들고, 기본 행동 대비 짝 차이가 z·표준오차보다 크면 가장 좋은 후보로 바꾼다
                starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
                for b0, b1 in zip(starts, np.r_[starts[1:], u]):
                    i = int(idx[b0])
                    qs = q[b0:b1].reshape(-1, s)  # 후보 순서: 기본 행동이 첫 줄
                    cands = first[b0:b1:s]
                    d = qs[1:] - qs[0]
                    mean = d.mean(1)
                    sem = d.std(1, ddof=1) / np.sqrt(s) if s > 1 else np.zeros_like(mean)
                    ok = (mean > 0) & (mean > z * sem)
                    stats["searched"] += 1
                    if ok.any():
                        j = int(np.argmax(np.where(ok, mean, -np.inf)))
                        c = int(cands[1 + j])
                        stats["changed"] += 1
                        stats["rank"][min(int((logits[i] > logits[i, c]).sum()), 4)] += 1
                        stats["gain"].append(float(mean[j]))
                        act[i] = c
        env.step(act, grid, scal, mask, rew, done, days)
        for f in env.pop_finished():
            i = f["env"]
            if alive[i]:
                final[i] = f["day"]
                alive[i] = False
        step += 1
    return final


def summary(name, fin, secs):
    r30 = fin >= 30
    score = np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin])
    return (f"{name}: day {fin.mean():.2f} · 점수 {score:.0f} · 30일 도달 {r30.mean():.1%} · "
            f"도달 후 통과 {(fin > 30).sum() / max(r30.sum(), 1):.1%} · 실행 {secs:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--k", type=int, default=3, help="확률 상위 후보 수")
    p.add_argument("--r", type=int, default=2, help="그 밖의 무작위 유효 행동 후보 수")
    p.add_argument("--s", type=int, default=4, help="가상 난수 묶음 수")
    p.add_argument("--z", type=float, default=2.0, help="기본 행동 대비 짝 차이 평균이 z·표준오차보다 클 때만 바꾼다 (0이면 평균만 비교)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    print(f"σ {pol.sigma:.3f} · γ {pol.gamma} · 후보 기본+상위 {args.k}+무작위 {args.r} · 난수 묶음 {args.s} · z {args.z}")

    t0 = time.time()
    base = play(pol, args.games, args.seed, False, 0, 0, 0, None)
    tb = time.time() - t0
    stats = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
    t0 = time.time()
    srch = play(pol, args.games, args.seed, True, args.k, args.r, args.s, stats, args.z)
    ts_ = time.time() - t0
    print(summary("기본", base, tb))
    print(summary("탐색", srch, ts_))
    d = srch - base
    print(f"짝 비교(탐색 - 기본): 생존일 {d.mean():+.2f} · 늘어남 {np.mean(d > 0):.0%} · 같음 {np.mean(d == 0):.0%} · 줄어듦 {np.mean(d < 0):.0%}")
    n = stats["searched"]
    print(f"탐색한 결정 {n} (게임당 {n / args.games:.0f}) · 기본 행동과 다른 선택 {stats['changed'] / max(n, 1):.1%} · "
          f"바뀐 선택의 원래 확률 순위(0=최상위) {stats['rank']} · 바뀐 선택의 평균 Q 이득 {np.mean(stats['gain']) if stats['gain'] else 0:.3f}")
    print(f"게임당 실행시간: 기본 {tb / args.games:.2f}s, 탐색 {ts_ / args.games:.2f}s (병렬 {args.games}판 기준)")


if __name__ == "__main__":
    main()
