"""증류 막힘 진단 (진단 전용: 게임 평가·실전 모델 교체에 쓰지 않는다). 원본 cn7_nf 20M에서 두 진단 모델을 새로 시작한다.
교사 표본: 기존 증류 데이터(distill_data.py 결과)에서 교사 한 수·준비·실제 완성을 고르게, 원본 정답 로그 확률의 분위를 고르게 뽑는다
  (매우 낮은 확률 포함). 학습 = 학습 게임(게임 % 10 != 9)에서 --n-train개, 검증 = 검증 게임에서 같은 방식으로 --n-val개.
T: 교사 표본만 교차 엔트로피(배치 안 교사 표본 평균, 계수 1).
M: T와 같은 교사 배치·순서·노출 횟수·계수 1 + λ × 기존 행동 유지 손실. 유지 손실 = 학습 게임 '유지' 상태(교사가 평가하고 정책 선택을 유지)의
   실제로 둔 행동의 교차 엔트로피, 별도 배치 --keep-bs개의 평균. 손실 = CE_교사(128개 평균) + λ · CE_유지(128개 평균).
KL·손실 상한·보상 변경 없음. gradient clipping도 끈다(전체 norm으로 자르면 M의 유지 기울기가 교사 기울기까지 줄이므로).
기록(LOG마다): 학습·검증 교사 표본의 분류별 정답 순위 중앙값·최빈 일치·NLL, 검증 유지 상태의 원본 최빈 선택 유지율·둔 행동 NLL·엔트로피
  (원본과의 KL은 기록만), M은 교사·유지 기울기 norm과 둘 사이 코사인.
사용: python rl/distill_tm.py [--updates 2000]"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from distill_student import NAMES, load, to_dev
from ppo import Agent

CATS = (1, 2, 3)


def net_from(ck, dev):
    a = ck["args"]
    m = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    m.load_state_dict(ck["agent"])
    return m


@torch.no_grad()
def orig_logp(orig, D, idx, dev, bs=4096):
    out = []
    for j in range(0, len(idx), bs):
        b = idx[j:j + bs]
        g, s, m = to_dev(D, b, dev)
        lp = F.log_softmax(orig(g, s)[0].masked_fill(~m, -1e9), 1)
        out.append(lp.gather(1, torch.from_numpy(D["act"][b]).long().to(dev)[:, None])[:, 0].cpu().numpy())
    return np.concatenate(out)


def pick(orig, D, pool, n, dev):
    """분류마다 n/3개, 원본 정답 로그 확률 분위를 고르게"""
    per = [n // 3 + (1 if k < n % 3 else 0) for k in range(3)]
    sel, info = [], {}
    for c, k in zip(CATS, per):
        cand = pool[D["cat"][pool] == c]
        lp = orig_logp(orig, D, cand, dev)
        o = np.argsort(lp)
        take = o[np.linspace(0, len(o) - 1, k).round().astype(int)]
        sel.append(cand[take])
        info[NAMES[c]] = {"후보": int(len(cand)), "고름": int(k), "원본 log p 최소": float(lp.min()), "중앙": float(np.median(lp[take])), "최대": float(lp.max())}
    return np.concatenate(sel), info


@torch.no_grad()
def evaluate(net, orig, D, idx, dev, bs=4096):
    acc = {}
    for j in range(0, len(idx), bs):
        b = idx[j:j + bs]
        g, s, m = to_dev(D, b, dev)
        lo = net(g, s)[0].masked_fill(~m, -1e9)
        lo0 = orig(g, s)[0].masked_fill(~m, -1e9)
        ls, l0 = F.log_softmax(lo, 1), F.log_softmax(lo0, 1)
        a = torch.from_numpy(D["act"][b]).long().to(dev)
        r = {"lp": ls.gather(1, a[:, None])[:, 0], "rank": (lo > lo.gather(1, a[:, None])).sum(1).float(), "top": (lo.argmax(1) == a).float(),
             "keep": (lo.argmax(1) == lo0.argmax(1)).float(), "ent": -(ls.exp() * ls.masked_fill(~m, 0)).sum(1),
             "kl": (l0.exp() * (l0 - ls)).masked_fill(~m, 0).sum(1)}
        cat = D["cat"][b]
        for c in np.unique(cat):
            sel = torch.from_numpy(cat == c).to(dev)
            e = acc.setdefault(int(c), {k: [] for k in r})
            for k, v in r.items():
                e[k].append(v[sel].cpu().numpy())
    out = {}
    for c, e in acc.items():
        e = {k: np.concatenate(v) for k, v in e.items()}
        out[NAMES[c]] = {"n": int(len(e["lp"])), "NLL": float(-e["lp"].mean()), "순위 중앙": float(np.median(e["rank"])), "최빈 일치": float(e["top"].mean()),
                         "원본 최빈 유지": float(e["keep"].mean()), "엔트로피": float(e["ent"].mean()), "KL(기록만)": float(e["kl"].mean())}
    return out


def fmt(ev):
    return " · ".join(f"{k} 순위 {ev[k]['순위 중앙']:.0f}·일치 {ev[k]['최빈 일치']:.0%}·NLL {ev[k]['NLL']:.2f}" for k in ("교사 한 수", "교사 준비", "실제 완성") if k in ev)


def teacher_total(ev):
    n = sum(ev[NAMES[c]]["n"] for c in CATS)
    return {k: sum(ev[NAMES[c]][k] * ev[NAMES[c]]["n"] for c in CATS) / n for k in ("NLL", "최빈 일치")}


def grads(net, loss):
    g = torch.autograd.grad(loss, [p for p in net.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
    return torch.cat([x.reshape(-1) if x is not None else torch.zeros(p.numel(), device=loss.device) for x, p in zip(g, net.parameters())])


def train(name, lam, ck, orig, D, tr, keep_tr, ev_sets, args, dev):
    torch.manual_seed(args.seed)
    net = net_from(ck, dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    rng_keep = np.random.default_rng([args.seed, 3])
    keep_order = rng_keep.permutation(keep_tr)
    kpos = 0
    run_dir = os.path.join("runs", f"distill_{name}")
    os.makedirs(run_dir, exist_ok=True)
    hist = []

    def log(u, extra=None):
        net.eval()
        ev = {k: evaluate(net, orig, D, idx, dev) for k, idx in ev_sets.items()}
        hist.append({"갱신": u, **ev, **(extra or {})})
        tt, tv = teacher_total(ev["학습 교사"]), teacher_total(ev["검증 교사"])
        k = ev["검증 유지"]["유지"]
        print(f"  [{name} {u:>4d}] 학습 교사 일치 {tt['최빈 일치']:.1%}·NLL {tt['NLL']:.2f} · 검증 교사 일치 {tv['최빈 일치']:.1%}·NLL {tv['NLL']:.2f} · "
              f"검증 유지: 원본 최빈 유지 {k['원본 최빈 유지']:.1%}·둔 행동 NLL {k['NLL']:.2f}·엔트로피 {k['엔트로피']:.2f}·KL {k['KL(기록만)']:.3f}"
              + (f" · {extra['기울기']}" if extra and "기울기" in extra else ""), flush=True)
        if u % 250 == 0 or u == 0:
            print(f"      학습: {fmt(ev['학습 교사'])}\n      검증: {fmt(ev['검증 교사'])}", flush=True)

    log(0)
    t0 = time.time()
    n_ep = 0
    order = np.random.default_rng([args.seed, 1, n_ep]).permutation(tr)
    pos = 0
    for u in range(1, args.updates + 1):
        if pos + args.bs > len(order):  # 에폭마다 새 순서 (T·M 같은 순서)
            n_ep += 1
            order = np.random.default_rng([args.seed, 1, n_ep]).permutation(tr)
            pos = 0
        b = order[pos:pos + args.bs]
        pos += args.bs
        net.train()
        g, s, m = to_dev(D, b, dev)
        a = torch.from_numpy(D["act"][b]).long().to(dev)
        ce_t = -F.log_softmax(net(g, s)[0].masked_fill(~m, -1e9), 1).gather(1, a[:, None])[:, 0].mean()
        loss = ce_t
        extra = None
        if lam > 0:
            if kpos + args.keep_bs > len(keep_order):
                keep_order = rng_keep.permutation(keep_tr)
                kpos = 0
            kb = keep_order[kpos:kpos + args.keep_bs]
            kpos += args.keep_bs
            gk, sk, mk = to_dev(D, kb, dev)
            ak = torch.from_numpy(D["act"][kb]).long().to(dev)
            ce_k = -F.log_softmax(net(gk, sk)[0].masked_fill(~mk, -1e9), 1).gather(1, ak[:, None])[:, 0].mean()
            loss = ce_t + lam * ce_k
            if u % args.log_every == 0:
                gt, gkk = grads(net, ce_t), grads(net, lam * ce_k)
                extra = {"기울기": f"교사 norm {gt.norm():.2f} · 유지 norm {gkk.norm():.2f} · 코사인 {F.cosine_similarity(gt, gkk, 0):+.3f}",
                         "손실 유지": float(ce_k)}
        opt.zero_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        if u % args.log_every == 0:
            extra = {**(extra or {}), "손실 교사": float(ce_t), "에폭": n_ep, "시간": round(time.time() - t0, 1)}
            log(u, extra)
        if u in (500, 1000, 2000) or u == args.updates:
            torch.save({"agent": net.state_dict(), "args": ck["args"], "rstd": ck["rstd"],
                        "diag": {"name": name, "lambda": lam, "updates": u, "lr": args.lr}}, os.path.join(run_dir, f"agent_u{u}.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "lambda": lam, "hist": hist}, f, ensure_ascii=False, indent=1)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--data", default="runs/distill_s12345.npz,runs/distill_s777.npz")
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-val", type=int, default=1024)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--keep-bs", type=int, default=128)
    p.add_argument("--lam", type=float, default=1.0, help="M의 유지 손실 계수")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", default="T,M", help="T, M, 또는 보조 Tc(T + gradient clipping --clip)")
    p.add_argument("--clip", type=float, default=0.0, help="gradient clipping norm (기본 끔; 보조 확인용)")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    D = load(args.data.split(","))
    val = D["game"] % 10 == 9
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    orig = net_from(ck, dev)
    orig.eval()
    for q in orig.parameters():
        q.requires_grad_(False)
    teach = np.isin(D["cat"], CATS)
    tr, info_tr = pick(orig, D, np.nonzero(teach & ~val)[0], args.n_train, dev)
    va, info_va = pick(orig, D, np.nonzero(teach & val)[0], args.n_val, dev)
    gid = D["src"] * 100000 + D["game"]
    assert not set(gid[tr]) & set(gid[va]), "학습·검증 게임이 겹친다"
    print(f"학습 교사 표본 {len(tr)} (게임 {len(np.unique(gid[tr]))}판): {info_tr}")
    print(f"검증 교사 표본 {len(va)} (게임 {len(np.unique(gid[va]))}판, 학습과 겹치는 게임 0): {info_va}")
    for name, idx in (("학습", tr), ("검증", va)):
        _, _, m = to_dev(D, idx, dev)
        ok = m.gather(1, torch.from_numpy(D["act"][idx]).long().to(dev)[:, None])[:, 0]
        keys = [hashlib.sha1(D["grid"][i].tobytes() + D["scal"][i].tobytes() + D["mask"][i].tobytes()).hexdigest() for i in idx]
        lab = {}
        for k, i in zip(keys, idx):
            lab.setdefault(k, set()).add(int(D["act"][i]))
        print(f"  {name}: 정답이 마스크에서 유효 {int(ok.sum())}/{len(idx)} · 같은 관측 묶음 {len(lab)} · 같은 관측에 다른 정답 {sum(len(v) > 1 for v in lab.values())}")
    keep_tr = np.nonzero((D["cat"] == 4) & ~val)[0]
    rng = np.random.default_rng([args.seed, 2])
    keep_va = rng.choice(np.nonzero((D["cat"] == 4) & val)[0], 4000, replace=False)
    ev_sets = {"학습 교사": tr, "검증 교사": va, "검증 유지": keep_va}
    print(f"T: 교사 CE(배치 {args.bs} 평균, 계수 1) · M: 같은 교사 배치 + λ={args.lam} × 유지 CE(학습 게임 유지 상태 {len(keep_tr)}개에서 배치 {args.keep_bs} 평균, 둔 행동) · "
          f"갱신 {args.updates} = 교사 표본마다 {args.updates * args.bs / len(tr):.0f}회 노출 · lr {args.lr} · clipping {args.clip or '없음'}", flush=True)
    for name in args.runs.split(","):
        train(name, args.lam if name == "M" else 0.0, ck, orig, D, tr, keep_tr, ev_sets, args, dev)


if __name__ == "__main__":
    main()
