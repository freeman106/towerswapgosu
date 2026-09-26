"""Tower Swap PPO (CleanRL 방식 단일 파일) + 행동 마스킹 + 하루 단위 할인.

할인은 날 단위다: 스텝 t에서 d_t일이 지나면(밤을 넘기면 1, 아니면 0) 그 스텝의 할인은 γ^d_t다.
따라서 하루 안의 행동끼리는 할인이 없다. GAE의 λ는 기본으로 행동마다 건다(--lambda-per-day 1이면 날 단위).

사용: python rl/ppo.py --run-name test --total-steps 5_000_000
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import towerswap as ts

C, H, W = ts.GRID_SHAPE
S = ts.N_SCALAR
A = ts.N_ACTIONS
N_DRAG, N_CELL = 168, 48
N_OTHER = A - N_DRAG - N_CELL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ppo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=128, help="환경당 롤아웃 길이")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.995, help="하루당 할인")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE 감쇠")
    p.add_argument("--lambda-per-day", type=int, default=0, help="1이면 λ도 날 단위, 0이면 행동 단위")
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--norm-reward", type=int, default=1, help="할인 누적 보상의 이동 표준편차로 보상을 나눈다")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=50, help="업데이트 단위")
    p.add_argument("--resume", default="", help="이어서 학습할 체크포인트")
    return p.parse_args()


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = layer_init(nn.Conv2d(c, c, 3, padding=1))
        self.c2 = layer_init(nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Agent(nn.Module):
    """격자(C×8×6)와 스칼라를 받아 225개 로짓과 가치를 낸다.
    드래그(42칸×4방향)와 칸 선택(48칸)은 칸마다 1×1 합성곱으로, 나머지 9개는 완전연결로 낸다."""

    def __init__(self, ch=64, blocks=4, hidden=256, s_emb=16):
        super().__init__()
        self.s_emb = layer_init(nn.Linear(S, s_emb))
        self.stem = layer_init(nn.Conv2d(C + s_emb, ch, 3, padding=1))
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.spatial = layer_init(nn.Conv2d(ch, 5, 1), std=0.01)  # 방향 4 + 칸 선택 1
        self.fc = layer_init(nn.Linear(ch * H * W, hidden))
        self.fc2 = layer_init(nn.Linear(hidden + S, hidden))
        self.other = layer_init(nn.Linear(hidden, N_OTHER), std=0.01)
        self.value = layer_init(nn.Linear(hidden, 1), std=1.0)

    def trunk(self, grid, scal):
        e = F.relu(self.s_emb(scal))[:, :, None, None].expand(-1, -1, H, W)
        x = F.relu(self.blocks(self.stem(torch.cat([grid, e], 1))))
        h = F.relu(self.fc(x.flatten(1)))
        h = F.relu(self.fc2(torch.cat([h, scal], 1)))
        return x, h

    def forward(self, grid, scal):
        x, h = self.trunk(grid, scal)
        sp = self.spatial(x)  # [B, 5, 8, 6]
        drag = sp[:, :4, 1:].permute(0, 2, 3, 1).reshape(-1, N_DRAG)  # ((y-1)*6 + x-1)*4 + dir
        cell = sp[:, 4].reshape(-1, N_CELL)  # y*6 + x-1
        logits = torch.cat([drag, cell, self.other(h)], 1)
        return logits, self.value(h).squeeze(1)

    def get_value(self, grid, scal):
        return self.value(self.trunk(grid, scal)[1]).squeeze(1)


class RunningStd:
    """할인 누적 보상의 분산을 추적한다 (gym NormalizeReward와 같은 방식, 할인은 날 단위)."""

    def __init__(self, n):
        self.ret = np.zeros(n, np.float64)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, rew, days, done, gamma):
        self.ret = self.ret * gamma ** days + rew
        x = self.ret
        bm, bv, bc = x.mean(), x.var(), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot
        self.ret[done] = 0.0

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


def masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
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
    env.reset(grid, scal, mask)

    agent = Agent(args.channels, args.blocks, args.hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_update, global_step = 1, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        agent.load_state_dict(ck["agent"])
        opt.load_state_dict(ck["opt"])
        start_update, global_step = ck["update"] + 1, ck["global_step"]
    print(f"장치 {dev}, 파라미터 {sum(p.numel() for p in agent.parameters()):,}개, 배치 {N}×{T}")

    b_grid = torch.zeros((T, N, C, H, W), device=dev)
    b_scal = torch.zeros((T, N, S), device=dev)
    b_mask = torch.zeros((T, N, A), dtype=torch.bool, device=dev)
    b_act = torch.zeros((T, N), dtype=torch.long, device=dev)
    b_logp = torch.zeros((T, N), device=dev)
    b_val = torch.zeros((T, N), device=dev)
    b_rew = torch.zeros((T, N), device=dev)
    b_done = torch.zeros((T, N), device=dev)
    b_days = torch.zeros((T, N), device=dev)

    n_updates = args.total_steps // (N * T)
    log_f = open(os.path.join(run_dir, "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    if start_update == 1:
        log.writerow(["update", "step", "sps", "episodes", "score_mean", "day_mean", "day_max", "bosses_mean",
                      "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "explained_var"])
    t_start, step0 = time.time(), global_step
    ep_hist = []
    rstd = RunningStd(N)
    for update in range(start_update, n_updates + 1):
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = args.lr * (1.0 - (update - 1.0) / n_updates)
        t_roll = time.time()
        # ── 롤아웃 ──
        for t in range(T):
            g = torch.from_numpy(grid).to(dev)
            s = torch.from_numpy(scal).to(dev)
            m = torch.from_numpy(mask).to(dev)
            b_grid[t], b_scal[t], b_mask[t] = g, s, m
            with torch.no_grad():
                logits, v = agent(g, s)
                dist = masked_dist(logits, m)
                a = dist.sample()
                b_act[t], b_logp[t], b_val[t] = a, dist.log_prob(a), v
            env.step(a.cpu().numpy(), grid, scal, mask, rew, done, days)
            if args.norm_reward:
                rstd.update(rew, days, done, args.gamma)
                b_rew[t] = torch.from_numpy(rew / rstd.std).to(dev)
            else:
                b_rew[t] = torch.from_numpy(rew).to(dev)
            b_done[t] = torch.from_numpy(done.astype(np.float32)).to(dev)
            b_days[t] = torch.from_numpy(days).to(dev)
        global_step += N * T
        t_roll = time.time() - t_roll

        # ── GAE (날 단위 할인) ──
        with torch.no_grad():
            next_v = agent.get_value(torch.from_numpy(grid).to(dev), torch.from_numpy(scal).to(dev))
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(N, device=dev)
            disc = args.gamma ** b_days
            lam = args.gae_lambda ** b_days if args.lambda_per_day else torch.full_like(b_days, args.gae_lambda)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + disc[t] * nv * nonterm - b_val[t]
                last = delta + disc[t] * lam[t] * nonterm * last
                adv[t] = last
            ret = adv + b_val

        # ── 학습 ──
        fg, fs, fm = b_grid.reshape(-1, C, H, W), b_scal.reshape(-1, S), b_mask.reshape(-1, A)
        fa, flp, fadv, fret, fval = b_act.reshape(-1), b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1), b_val.reshape(-1)
        bs = N * T
        mb = bs // args.minibatches
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(bs, device=dev)
            for i in range(0, bs, mb):
                idx = perm[i:i + mb]
                logits, v = agent(fg[idx], fs[idx])
                dist = masked_dist(logits, fm[idx])
                logp = dist.log_prob(fa[idx])
                ratio = (logp - flp[idx]).exp()
                a_mb = fadv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * ((v - fret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    lr_ = logp - flp[idx]
                    kl = ((ratio - 1) - lr_).mean()
                    cf = ((ratio - 1).abs() > args.clip).float().mean()
                stats.append(torch.stack([pg.detach(), v_loss.detach(), ent.detach(), kl, cf]))
        st = torch.stack(stats).mean(0).tolist()
        var_y = fret.var()
        ev = (1 - (fret - fval).var() / var_y).item() if var_y > 0 else float("nan")

        # ── 기록 ──
        fin = env.pop_finished()
        ep_hist = (ep_hist + fin)[-2000:]
        sps = (global_step - step0) / (time.time() - t_start)
        if fin:
            sc = np.array([f[0] for f in fin])
            dy = np.array([f[1] for f in fin])
            row = [update, global_step, int(sps), len(fin), sc.mean(), dy.mean(), dy.max(), (sc // 1000).mean()] + st + [ev]
        else:
            row = [update, global_step, int(sps), 0, "", "", "", ""] + st + [ev]
        log.writerow(row)
        log_f.flush()
        if update % 5 == 0 or update == 1:
            h = np.array([f[1] for f in ep_hist]) if ep_hist else np.zeros(1)
            print(f"[{update}/{n_updates}] step {global_step:,} sps {sps:,.0f} (롤아웃 {t_roll:.1f}s) | "
                  f"최근 {len(ep_hist)}게임 day 평균 {h.mean():.2f} 최대 {h.max()} | "
                  f"pg {st[0]:.4f} v {st[1]:.4f} ent {st[2]:.3f} kl {st[3]:.4f} ev {ev:.3f} rstd {rstd.std:.4f}", flush=True)
        if update % args.save_every == 0 or update == n_updates:
            torch.save({"agent": agent.state_dict(), "opt": opt.state_dict(), "update": update,
                        "global_step": global_step, "args": vars(args)}, os.path.join(run_dir, "agent.pt"))


if __name__ == "__main__":
    main()
