"""탐색 증류: search_eval.py --record 데이터로 학생을 학습한다(가중치는 --init 체크포인트에서 시작).
- 정책: 탐색이 고른 후보 b와 통계적으로 구분되지 않는 후보 집합 T에 확률을 모은다: −log Σ_{a∈T} π(a|s).
  T = {c : 차이평균_c ≥ 차이평균_b − z·√(표준오차_b² + 표준오차_c²)} (차이는 기본 행동 대비, 같은 난수 묶음의 짝 차이).
  확실히 이긴 선택은 T가 하나뿐이라 강하게, 동점권은 집합 전체를 허용한다. 탐색을 안 한 결정(유효 행동 1개)은 제외.
  --target mix: 목표 분포 p* = 바꾸지 않은 결정은 시작 정책 π₀ 그대로, 바꾼 결정은 (1−α)·π₀ + α·(탐색이 고른 행동).
  교차 엔트로피 −Σ p* log π로 학습해 원래 행동은 유지하고 탐색이 바꾼 곳만 움직인다(보수적 정책 개선).
  --target awr: p* ∝ π₀·exp(Â/τ), Â는 후보별 기본 대비 짝 차이 평균(후보가 아닌 행동은 π₀ 비율 유지).
- 가치: 탐색 정책이 실제로 얻은 할인 수익 G_t = r_t/σ + γ^일수·G_{t+1} (σ, γ는 데이터의 값으로 고정).
사용: python rl/distill.py rl/runs/cb9/agent.pt rl/runs/distill_data1.npz --run-name ds1 [--epochs 4] [--z 2]"""
import argparse
import os

import numpy as np
import torch

from ppo import A, Agent


def load(paths):
    parts = [np.load(p) for p in paths.split(",")]
    d = {}
    offset = 0
    for P in parts:
        game = P["game"] + offset
        offset = game.max() + 1
        for k in ("grid", "scal", "mask", "act", "cands", "qmean", "dmean", "dsem", "searched", "rew", "days", "done"):
            d.setdefault(k, []).append(P[k])
        d.setdefault("game", []).append(game)
    d = {k: np.concatenate(v) for k, v in d.items()}
    d["sigma"], d["gamma"] = float(parts[0]["sigma"]), float(parts[0]["gamma"])
    return d


def returns(d):
    """게임별 할인 수익 (행은 게임마다 시간 순서)"""
    G = np.zeros(len(d["rew"]), np.float32)
    for g in np.unique(d["game"]):
        run = 0.0
        for j in np.nonzero(d["game"] == g)[0][::-1]:
            run = d["rew"][j] / d["sigma"] + (0.0 if d["done"][j] else d["gamma"] ** d["days"][j] * run)
            G[j] = run
    return G


def target_sets(d, z):
    """정책 목표 집합 [N, A] (탐색하지 않은 행은 전부 거짓)"""
    n = len(d["act"])
    T = np.zeros((n, A), bool)
    for j in np.nonzero(d["searched"])[0]:
        c, dm, ds = d["cands"][j], d["dmean"][j], d["dsem"][j]
        ok = c >= 0
        b = int(np.nonzero(c == d["act"][j])[0][0])
        thr = dm[b] - z * np.sqrt(ds[b] ** 2 + ds ** 2)
        T[j, c[ok & (dm >= thr)]] = True
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("init")
    p.add_argument("data", help="npz 경로 (쉼표로 여러 개)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--z", type=float, default=2.0, help="목표 집합의 동점 허용 폭(표준오차 배수)")
    p.add_argument("--target", default="set", choices=["set", "mix", "awr"])
    p.add_argument("--tau", type=float, default=0.02, help="awr: 목표 p* ∝ π₀·exp(이득/τ) (후보만 재가중, 이득 = 기본 대비 짝 차이 평균)")
    p.add_argument("--alpha", type=float, default=0.5, help="mix: 바꾼 결정에서 탐색 선택에 줄 확률 비중")
    p.add_argument("--min-gain", type=float, default=0.0, help="mix: 이 이득 이하의 교체는 무시(시작 정책 유지)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--pol-coef", type=float, default=1.0, help="0이면 정책은 그대로 두고 가치망만 학습(정책 쪽 층은 고정)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    d = load(args.data)
    G = returns(d)
    T = target_sets(d, args.z)
    mask = np.unpackbits(d["mask"], axis=1)[:, :A].astype(bool)
    searched = d["searched"]
    base_changed = np.array([searched[j] and d["cands"][j][0] != d["act"][j] for j in range(len(searched))])
    if args.min_gain > 0:  # 이득이 작은 교체는 바꾸지 않은 결정으로 본다
        gain = np.array([d["dmean"][j][np.nonzero(d["cands"][j] == d["act"][j])[0][0]] if base_changed[j] else 0.0 for j in range(len(searched))])
        base_changed &= gain > args.min_gain
    W = np.ones((len(searched), A), np.float32)  # awr 재가중치 exp(Â/τ)
    if args.target == "awr":
        for j in np.nonzero(searched)[0]:
            c = d["cands"][j]
            ok = c >= 0
            W[j, c[ok]] = np.exp(np.clip(d["dmean"][j][ok] / args.tau, -20, 20))
    games = np.unique(d["game"])
    val_games = games[-max(1, int(len(games) * args.val_frac)):]
    val = np.isin(d["game"], val_games)
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(f"데이터 {len(G)}행 (게임 {len(games)}, 검증 {len(val_games)}) · 탐색한 결정 {searched.sum()} · 바뀐 결정 {base_changed.sum()} · "
          f"목표 집합 평균 크기 {T[searched].sum(1).mean():.2f} · 하나뿐인 비율 {(T[searched].sum(1) == 1).mean():.1%} · "
          f"수익 평균 {G.mean():.2f} (σ {d['sigma']:.3f}, γ {d['gamma']})", flush=True)

    ck = torch.load(args.init, map_location=dev, weights_only=False)
    a = ck["args"]
    agent = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    agent.load_state_dict(ck["agent"])
    if args.pol_coef == 0:  # 정책 출력 층을 고정: 공유 몸통이 바뀌면 정책도 바뀌므로, 가치 머리만 학습한다
        for name, prm in agent.named_parameters():
            prm.requires_grad = name.startswith("value.")
    opt = torch.optim.Adam([q for q in agent.parameters() if q.requires_grad], lr=args.lr, eps=1e-5)
    ref = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)  # 시작 정책 π₀ (고정)
    ref.load_state_dict(ck["agent"])
    ref.eval()

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for b0 in range(0, len(idx), args.batch):
            j = np.sort(idx[b0:b0 + args.batch])
            yield (j, torch.from_numpy(d["grid"][j].astype(np.float32)).to(dev), torch.from_numpy(d["scal"][j].astype(np.float32)).to(dev),
                   torch.from_numpy(mask[j]).to(dev), torch.from_numpy(T[j]).to(dev), torch.from_numpy(G[j]).to(dev),
                   torch.from_numpy(searched[j]).to(dev))

    def losses(g, s, m, t, ret, sr, j=None):
        logits, v = agent(g, s)
        lm = logits.masked_fill(~m, -1e9)
        if args.target == "set":
            pol = lm.logsumexp(1) - lm.masked_fill(~t, -1e9).logsumexp(1)
        else:
            with torch.no_grad():
                p0 = torch.softmax(ref(g, s)[0].masked_fill(~m, -1e9), 1)
                ch = torch.from_numpy(base_changed[j]).to(dev)
                onehot = torch.zeros_like(p0).scatter_(1, torch.from_numpy(d["act"][j]).to(dev)[:, None], 1.0)
                if args.target == "mix":
                    pt = torch.where(ch[:, None], (1 - args.alpha) * p0 + args.alpha * onehot, p0)
                else:
                    pt = p0 * torch.from_numpy(W[j]).to(dev)
                    pt = pt / pt.sum(1, keepdim=True)
            pol = -(pt * torch.log_softmax(lm, 1)).sum(1) + (pt * torch.log(pt.clamp(min=1e-12))).sum(1)  # KL(p* || π)
        pol = (pol * sr).sum() / sr.sum().clamp(min=1)
        vl = 0.5 * ((v - ret) ** 2).mean()
        hit = (t.gather(1, lm.argmax(1, keepdim=True)).squeeze(1) & sr).sum() / sr.sum().clamp(min=1)
        return pol, vl, hit, lm

    def evaluate(idx):
        agent.eval()
        tot = np.zeros(4)
        n = 0
        with torch.no_grad():
            for j, g, s, m, t, ret, sr in batches(idx, False):
                pol, vl, hit, lm = losses(g, s, m, t, ret, sr, j)
                ch = torch.from_numpy(base_changed[j]).to(dev)
                chosen = torch.from_numpy(d["act"][j]).to(dev)
                ch_hit = ((lm.argmax(1) == chosen) & ch).sum() / ch.sum().clamp(min=1)
                w = len(j)
                tot += np.array([pol.item(), vl.item(), hit.item(), ch_hit.item()]) * w
                n += w
        agent.train()
        return tot / n

    r = evaluate(va_idx)
    print(f"[시작] 검증: 정책 손실 {r[0]:.3f} · 가치 손실 {r[1]:.3f} · 최빈 행동이 목표 집합 안 {r[2]:.1%} · 바뀐 결정에서 탐색 선택과 일치 {r[3]:.1%}", flush=True)
    for ep in range(1, args.epochs + 1):
        tl = []
        for j, g, s, m, t, ret, sr in batches(tr_idx, True):
            pol, vl, hit, _ = losses(g, s, m, t, ret, sr, j)
            loss = args.pol_coef * pol + args.vf_coef * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()
            tl.append([pol.item(), vl.item(), hit.item()])
        tl = np.mean(tl, 0)
        r = evaluate(va_idx)
        print(f"[{ep}] 학습: 정책 {tl[0]:.3f} · 가치 {tl[1]:.3f} · 집합 안 {tl[2]:.1%} | 검증: 정책 {r[0]:.3f} · 가치 {r[1]:.3f} · "
              f"집합 안 {r[2]:.1%} · 바뀐 결정 일치 {r[3]:.1%}", flush=True)
        torch.save({"agent": agent.state_dict(), "args": a, "rstd": ck["rstd"], "distill": vars(args), "epoch": ep},
                   os.path.join(run_dir, f"agent_e{ep}.pt"))
    torch.save({"agent": agent.state_dict(), "args": a, "rstd": ck["rstd"], "distill": vars(args)}, os.path.join(run_dir, "agent.pt"))


if __name__ == "__main__":
    main()
