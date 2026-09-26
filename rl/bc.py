"""greedy 전문가 모방 학습 (DAgger 방식, 온라인). 결과는 ppo.py --init으로 이어서 학습한다.

매 스텝 전문가의 최선 행동 집합(VecEnv.expert, 휴리스틱 동점 전부)을 라벨로 기록하고, 그 집합에 실린 확률의 합을 키운다. 게임은 확률 β로 전문가, 1-β로 학생 정책이 진행하며
β는 1에서 0으로 줄어든다. 가치 머리도 진행 정책의 날 단위 할인 수익(GAE 목표)으로 함께 학습한다.
관측이 커서(샘플당 약 6.5KB) 데이터를 쌓지 않고 롤아웃마다 새로 모은다.

사용: python rl/bc.py --run-name bc1 --total-steps 20_000_000
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

import towerswap as ts
from ppo import A, C, H, S, W, Agent, RunningStd, masked_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="bc")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--beta-frac", type=float, default=0.5, help="β가 0이 되는 시점(전체 스텝 비율)")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    N, T = args.num_envs, args.num_steps
    env = ts.VecEnv(N, seed=args.seed, threads=args.threads)
    grid = np.zeros((N, C, H, W), np.float32)
    scal = np.zeros((N, S), np.float32)
    mask = np.zeros((N, A), bool)
    rew = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    days = np.zeros(N, np.float32)
    exp_act = np.zeros(N, np.int64)
    exp_set = np.zeros((N, A), bool)
    env.reset(grid, scal, mask)

    agent = Agent(args.channels, args.blocks, args.hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    rstd = RunningStd(N)
    print(f"장치 {dev}, 파라미터 {sum(p.numel() for p in agent.parameters()):,}개, 배치 {N}×{T}")

    b_grid = torch.zeros((T, N, C, H, W), device=dev)
    b_scal = torch.zeros((T, N, S), device=dev)
    b_mask = torch.zeros((T, N, A), dtype=torch.bool, device=dev)
    b_set = torch.zeros((T, N, A), dtype=torch.bool, device=dev)
    b_val = torch.zeros((T, N), device=dev)
    b_rew = torch.zeros((T, N), device=dev)
    b_done = torch.zeros((T, N), device=dev)
    b_days = torch.zeros((T, N), device=dev)

    n_iters = args.total_steps // (N * T)
    log_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "step", "sps", "beta", "ce", "acc", "v_loss", "episodes", "score_mean", "day_mean", "day_max"])
    t0, global_step, ep_hist = time.time(), 0, []
    for it in range(1, n_iters + 1):
        beta = max(0.0, 1.0 - global_step / (args.beta_frac * args.total_steps))
        opt.param_groups[0]["lr"] = args.lr * (1.0 - (it - 1.0) / n_iters)
        for t in range(T):
            g = torch.from_numpy(grid).to(dev)
            s = torch.from_numpy(scal).to(dev)
            m = torch.from_numpy(mask).to(dev)
            b_grid[t], b_scal[t], b_mask[t] = g, s, m
            with torch.no_grad():
                logits, v = agent(g, s)
                student = masked_dist(logits, m).sample()
            env.expert(exp_act, exp_set)  # GPU 계산과 겹친다
            drive = np.where(rng.random(N) < beta, exp_act, student.cpu().numpy())
            b_set[t] = torch.from_numpy(exp_set).to(dev)
            b_val[t] = v
            env.step(drive, grid, scal, mask, rew, done, days)
            rstd.update(rew, days, done, args.gamma)
            b_rew[t] = torch.from_numpy(rew / rstd.std).to(dev)
            b_done[t] = torch.from_numpy(done.astype(np.float32)).to(dev)
            b_days[t] = torch.from_numpy(days).to(dev)
        global_step += N * T

        with torch.no_grad():  # 진행 정책의 가치 목표 (ppo.py와 같은 GAE)
            next_v = agent.get_value(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(N, device=dev)
            disc = args.gamma ** b_days
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + disc[t] * nv * nonterm - b_val[t]
                last = delta + disc[t] * args.gae_lambda * nonterm * last
                adv[t] = last
            ret = (adv + b_val).reshape(-1)

        fg, fs, fm, fe = b_grid.reshape(-1, C, H, W), b_scal.reshape(-1, S), b_mask.reshape(-1, A), b_set.reshape(-1, A)
        bs = N * T
        mb = bs // args.minibatches
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(bs, device=dev)
            for i in range(0, bs, mb):
                idx = perm[i:i + mb]
                logits, v = agent(fg[idx], fs[idx])
                logits = logits.masked_fill(~fm[idx], -1e9)
                # 최선 행동 집합(동점 전부)에 실린 확률의 합을 키운다
                ce = (logits.logsumexp(1) - logits.masked_fill(~fe[idx], -1e9).logsumexp(1)).mean()
                v_loss = 0.5 * ((v - ret[idx]) ** 2).mean()
                loss = ce + args.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                acc = fe[idx].gather(1, logits.argmax(1, keepdim=True)).float().mean()
                stats.append(torch.stack([ce.detach(), acc, v_loss.detach()]))
        ce, acc, vl = torch.stack(stats[-args.minibatches:]).mean(0).tolist()  # 마지막 에폭

        fin = env.pop_finished()
        ep_hist = (ep_hist + fin)[-1000:]
        sps = global_step / (time.time() - t0)
        if fin:
            sc, dy = np.array([f[0] for f in fin]), np.array([f[1] for f in fin])
            log.writerow([it, global_step, int(sps), beta, ce, acc, vl, len(fin), sc.mean(), dy.mean(), dy.max()])
        else:
            log.writerow([it, global_step, int(sps), beta, ce, acc, vl, 0, "", "", ""])
        log_f.flush()
        if it % 10 == 0 or it == 1:
            h = np.array([f[1] for f in ep_hist]) if ep_hist else np.zeros(1)
            sc = np.array([f[0] for f in ep_hist]) if ep_hist else np.zeros(1)
            print(f"[{it}/{n_iters}] step {global_step:,} sps {sps:,.0f} β {beta:.2f} | ce {ce:.3f} acc {acc:.3f} v {vl:.3f} | "
                  f"최근 {len(ep_hist)}게임 점수 {sc.mean():.0f} day 평균 {h.mean():.2f} 최대 {h.max()}", flush=True)
        if it % args.save_every == 0 or it == n_iters:
            torch.save({"agent": agent.state_dict(), "args": vars(args),
                        "rstd": (rstd.mean, rstd.var, rstd.count)}, os.path.join(run_dir, "agent.pt"))


if __name__ == "__main__":
    main()
