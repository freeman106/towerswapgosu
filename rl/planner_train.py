"""계획 선택기 학습 (원본 정책·인코더 동결, 선택기 MLP만 학습). 데이터 = planner_data.py 결과(교사가 평가한 모든 결정).
분할(게임 단위, 한 게임의 모든 상태·후보는 같은 쪽): 게임 % 10 != 9 학습, 검증 중 게임 % 20 == 9는 문턱 고르기·조기 종료용, % 20 == 19는 보고용.
손실 = 통과 BCE(모든 결정의 모든 후보: 기록 안 된 결정은 교사가 유지했으므로 전부 불통과, 선택 편향 없음)
     + ΔQ 가중 MSE(ΔQ를 아는 후보만: 교체 결정 가중치 1, 기록된 유지 결정 가중치 25 = 기록 확률 1/25의 역수, 표준화 단위).
ΔQ를 모르는 후보는 ΔQ 손실에서 뺀다(0으로 채우지 않는다).
선택 규칙: 통과 확률 ≥ τ인 후보가 있으면 그중 예측 ΔQ 최대, 없으면 원본 선택. τ는 문턱 세트에서 ΔQ 이득/결정(IPW: 기록된 결정의 실제 ΔQ,
  유지 기록 가중치 25)이 최대인 값(전체 데이터 학습 전에 정함). 참고로 (통과 개입 − 불통과 개입) 최대 τ도 보인다.
보고(보고 세트): 개입률, 개입 중 교사 기준 통과 비율, 교사 교체 결정 중 찾아낸 비율(통과 후보로 개입)·같은 후보 비율, 비교 기준(항상 유지, 항상 첫 후보).
사용: python rl/planner_train.py [--out runs/planner_sel]"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from planner import NC, Selector, features

KEYS = ("grid", "scal", "game", "day", "hearts", "swaps", "base", "lp_base", "b_ok", "b_same", "b_sig", "b_cr", "b_lost", "b_res", "b_dsw",
        "k", "a1", "a2", "lp1", "c_sig", "c_cr", "c_lost", "c_res", "passed", "dq", "dse", "logged", "replaced", "chosen")


def load(paths):
    parts = []
    for s, p in enumerate(paths):
        z = np.load(p)
        d = {k: z[k] for k in KEYS}
        d["src"] = np.full(len(d["k"]), s)
        parts.append(d)
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def select(p, q, valid, tau):
    """통과 확률 p ≥ τ인 후보 중 예측 ΔQ 최대 (없으면 -1)"""
    ok = valid & (p >= tau)
    c = np.where(ok, q, -np.inf).argmax(1)
    return np.where(ok.any(1), c, -1)


def sel_metrics(c, D, idx, keep_w=25.0):
    passed, rep, chosen = D["passed"][idx], D["replaced"][idx], D["chosen"][idx]
    iv = c >= 0
    good = iv & (np.take_along_axis(passed, np.maximum(c, 0)[:, None], 1)[:, 0] == 1)
    n_rep = int(rep.sum())
    # ΔQ 이득(IPW): 기록된 결정만 실제 ΔQ를 안다. 교체 결정은 전부(가중치 1), 유지 결정은 1/25만 기록(가중치 25)
    w = np.where(D["logged"][idx], np.where(rep, 1.0, keep_w), 0.0)
    dq_c = np.nan_to_num(np.take_along_axis(D["dq"][idx], np.maximum(c, 0)[:, None], 1)[:, 0]) * iv
    dq_t = np.nan_to_num(np.take_along_axis(D["dq"][idx], np.maximum(chosen, 0)[:, None], 1)[:, 0]) * (chosen >= 0)
    return {"결정": int(len(idx)), "개입률": float(iv.mean()), "교사 교체율": float(rep.mean()), "개입": int(iv.sum()),
            "개입 중 교사 기준 통과": float(good.sum() / max(iv.sum(), 1)), "교사 교체 중 찾아냄": float(good[rep].sum() / max(n_rep, 1)),
            "교사 교체 중 같은 후보": float((iv & (c == chosen))[rep].sum() / max(n_rep, 1)), "통과 − 불통과 개입": int(good.sum() - (iv & ~good).sum()),
            "ΔQ 이득/결정(IPW)": float((w * dq_c).sum() / max(w.sum(), 1e-9)), "교사 ΔQ 이득/결정(IPW)": float((w * dq_t).sum() / max(w.sum(), 1e-9))}


@torch.no_grad()
def predict(m, D, H, CL, idx, dev, bs=2048):
    P, Q = [], []
    for j in range(0, len(idx), bs):
        b = idx[j:j + bs]
        g = torch.from_numpy(D["grid"][b].astype(np.float32)).to(dev)
        s = torch.from_numpy(D["scal"][b].astype(np.float32)).to(dev)
        lp, dq = m(g, s, torch.from_numpy(H[b]).to(dev), torch.from_numpy(CL[b]).to(dev))
        P.append(torch.sigmoid(lp).cpu().numpy())
        Q.append(dq.cpu().numpy())
    return np.concatenate(P), np.concatenate(Q)


def losses(m, D, H, CL, V, W, dq_std, idx, dev, bs=2048):
    tot, n = np.zeros(2), 0
    p, q = predict(m, D, H, CL, idx, dev, bs)
    v = V[idx]
    y = D["passed"][idx] == 1
    pp = np.clip(p, 1e-6, 1 - 1e-6)
    bce = -(y * np.log(pp) + (1 - y) * np.log(1 - pp))[v].mean()
    w = W[idx]
    t = np.nan_to_num(D["dq"][idx] / dq_std)
    mse = (w * (q - t) ** 2).sum() / max(w.sum(), 1e-9)
    return float(bce), float(mse), p, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/cn7_nf/agent_20M.pt")
    ap.add_argument("--data", default="runs/planner_s12345.npz,runs/planner_s777.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--keep-w", type=float, default=25.0, help="기록된 유지 결정의 ΔQ 가중치 (기록 확률의 역수)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/planner_sel")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    D = load(args.data.split(","))
    H, CL, V = features(D)
    g = D["game"]
    tr = np.nonzero(g % 10 != 9)[0]
    th = np.nonzero(g % 20 == 9)[0]
    rp = np.nonzero(g % 20 == 19)[0]
    gid = D["src"] * 100000 + g
    assert not (set(gid[tr]) & set(gid[th])) and not (set(gid[tr]) & set(gid[rp])) and not (set(gid[th]) & set(gid[rp]))
    known = ~np.isnan(D["dq"]) & V
    W = np.where(known, np.where(D["replaced"][:, None], 1.0, args.keep_w), 0.0).astype(np.float32)
    dq_std = float(np.sqrt(np.average(np.nan_to_num(D["dq"][tr]) ** 2, weights=W[tr] + 1e-12)))
    print(f"결정 {len(g)} (학습 {len(tr)}·문턱 {len(th)}·보고 {len(rp)}, 게임 {len(np.unique(gid[tr]))}/{len(np.unique(gid[th]))}/{len(np.unique(gid[rp]))}) · "
          f"후보 {int(V.sum())} · 통과 {int((D['passed'] == 1).sum())} · ΔQ 아는 후보 {int(known.sum())} (교체 결정 {int((known & D['replaced'][:, None]).sum())}, "
          f"기록된 유지 결정 {int((known & ~D['replaced'][:, None]).sum())} × 가중치 {args.keep_w:g}) · ΔQ 표준화 {dq_std:.3f} · 손 특징 {H.shape[-1]}", flush=True)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    m = Selector(ck["args"], H.shape[-1]).to(dev)
    m.enc.load_state_dict(ck["agent"])
    hv = H[tr][V[tr]]
    m.mu.copy_(torch.from_numpy(hv.mean(0)))
    m.sd.copy_(torch.from_numpy(hv.std(0) + 1e-3))
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=args.lr)
    rng = np.random.default_rng([args.seed, 1])
    os.makedirs(args.out, exist_ok=True)
    best, bad, hist = (np.inf, -1), 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        m.train()
        order = rng.permutation(tr)
        for j in range(0, len(order), args.bs):
            b = order[j:j + args.bs]
            gg = torch.from_numpy(D["grid"][b].astype(np.float32)).to(dev)
            ss = torch.from_numpy(D["scal"][b].astype(np.float32)).to(dev)
            lp, dq = m(gg, ss, torch.from_numpy(H[b]).to(dev), torch.from_numpy(CL[b]).to(dev))
            v = torch.from_numpy(V[b]).to(dev)
            y = torch.from_numpy((D["passed"][b] == 1).astype(np.float32)).to(dev)
            bce = F.binary_cross_entropy_with_logits(lp[v], y[v])
            w = torch.from_numpy(W[b]).to(dev)
            t = torch.from_numpy(np.nan_to_num(D["dq"][b] / dq_std).astype(np.float32)).to(dev)
            mse = (w * (dq - t) ** 2).sum() / w.sum().clamp(min=1e-9)
            loss = bce + mse
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        bce_t, mse_t, p, q = losses(m, D, H, CL, V, W, dq_std, th, dev)
        taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
        scores = [sel_metrics(select(p, q, V[th], t), D, th)["ΔQ 이득/결정(IPW)"] for t in taus]
        tau = float(taus[int(np.argmax(scores))])
        mt = sel_metrics(select(p, q, V[th], tau), D, th)
        hist.append({"에폭": ep, "문턱 세트 BCE": bce_t, "문턱 세트 ΔQ MSE": mse_t, "τ": tau, **mt})
        print(f"에폭 {ep} ({time.time() - t0:.0f}s) · 문턱 세트 BCE {bce_t:.4f} · ΔQ MSE {mse_t:.3f} · τ {tau} → 개입률 {mt['개입률']:.1%} "
              f"(교사 {mt['교사 교체율']:.1%}) · 개입 중 통과 {mt['개입 중 교사 기준 통과']:.1%} · 교사 교체 중 찾아냄 {mt['교사 교체 중 찾아냄']:.1%} · "
              f"ΔQ 이득/결정 {mt['ΔQ 이득/결정(IPW)']:+.4f} (교사 {mt['교사 ΔQ 이득/결정(IPW)']:+.4f})", flush=True)
        if bce_t + mse_t < best[0]:
            best, bad = (bce_t + mse_t, ep), 0
            torch.save({"state": m.state_dict(), "enc_args": ck["args"], "d_hand": H.shape[-1], "dq_std": dq_std, "epoch": ep}, os.path.join(args.out, "selector.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break
    # 최고 에폭으로 문턱을 정하고 보고 세트에서 평가
    sd = torch.load(os.path.join(args.out, "selector.pt"), map_location=dev, weights_only=False)
    m.load_state_dict(sd["state"])
    m.eval()
    _, _, p, q = losses(m, D, H, CL, V, W, dq_std, th, dev)
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tab = {float(t): sel_metrics(select(p, q, V[th], t), D, th) for t in taus}
    tau = max(tab, key=lambda t: tab[t]["ΔQ 이득/결정(IPW)"])
    tau_net = max(tab, key=lambda t: tab[t]["통과 − 불통과 개입"])
    bce_r, mse_r, pr, qr = losses(m, D, H, CL, V, W, dq_std, rp, dev)
    rep = {float(t): sel_metrics(select(pr, qr, V[rp], t), D, rp) for t in taus}
    first = np.where(D["k"][rp] > 0, 0, -1)
    always_first = sel_metrics(first, D, rp)
    keep = sel_metrics(np.full(len(rp), -1), D, rp)
    print(f"\n최고 에폭 {sd['epoch']} · 문턱 세트에서 정한 τ = {tau} (참고: 통과 − 불통과 최대 τ = {tau_net}) · 보고 세트 BCE {bce_r:.4f} · ΔQ MSE {mse_r:.3f}")
    print("τ별 (보고 세트): τ · 개입률 · 개입 중 통과 · 교사 교체 중 찾아냄 · 같은 후보 · 통과 − 불통과 · ΔQ 이득/결정(IPW)")
    for t in taus:
        r = rep[float(t)]
        print(f"  {t:.2f}{' ←' if float(t) == tau else '  '} {r['개입률']:6.1%} · {r['개입 중 교사 기준 통과']:6.1%} · {r['교사 교체 중 찾아냄']:6.1%} · "
              f"{r['교사 교체 중 같은 후보']:6.1%} · {r['통과 − 불통과 개입']:+d} · {r['ΔQ 이득/결정(IPW)']:+.4f}")
    print(f"  교사(보고 세트): 교체율 {rep[tau]['교사 교체율']:.1%} · ΔQ 이득/결정(IPW) {rep[tau]['교사 ΔQ 이득/결정(IPW)']:+.4f}")
    print(f"비교 기준(보고 세트): 항상 유지 {keep} \n  항상 첫 후보(교사 정렬 1번) {always_first}")
    sd["tau"], sd["tau_net"] = tau, tau_net
    torch.save(sd, os.path.join(args.out, "selector.pt"))
    with open(os.path.join(args.out, "train.json"), "w") as f:
        json.dump({"args": vars(args), "hist": hist, "tau": tau, "tau_net": tau_net, "threshold_table": tab, "report_table": rep, "always_first": always_first,
                   "always_keep": keep}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
