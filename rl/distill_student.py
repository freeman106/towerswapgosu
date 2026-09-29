"""교사 증류 학생 학습 (정책 모방만, PPO·보상 변경 없음). 원본 cn7_nf 20M 가중치에서 시작한 별도 학생을 만든다(원본 파일은 읽기만 한다).
데이터: distill_data.py 결과(실제 관측·환경 마스크·실제로 둔 행동). 게임 단위 분리: 게임 번호 % 10 == 9 는 검증, 나머지는 학습.
학습: 교사 한 수(1)·교사 두 수 준비(2)·실제 준비 뒤 관측의 실제 완성(3) + 교사가 기존 선택을 유지한 상태(4)를 자연 비율로 섞어,
  환경 마스크 안에서 실제로 둔 행동의 음의 로그 확률(교차 엔트로피). 정답만 남기는 강제 마스크는 쓰지 않는다.
  재확인 실패 뒤 상태(5)와 교사 평가가 없는 상태(0)는 학습하지 않고, 원본 정책과의 KL로 변화만 본다.
기록(학습·검증, 에폭마다): 분류별 손실·정답 확률·최빈 행동 일치율, 개입 상태에서 기존 선택(정책이 원래 고른 수)의 확률, 원본 대비 KL.
사용: python rl/distill_student.py --run student1 [--epochs 10 --lr 3e-5] [--frac 0.2 (소량 확인)]"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ppo import A, C, H, S, W, Agent

NAMES = {0: "그 밖", 1: "교사 한 수", 2: "교사 준비", 3: "실제 완성", 4: "유지", 5: "재확인 실패 뒤"}
TRAIN_CATS = (1, 2, 3, 4)


def load(paths):
    parts = []
    for i, p in enumerate(paths):
        z = np.load(p)
        d = {k: z[k] for k in ("grid", "scal", "mask", "act", "base", "cat", "game", "failed", "same")}
        d["src"] = np.full(len(d["act"]), i)
        parts.append(d)
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def batches(idx, bs, rng=None):
    if rng is not None:
        idx = rng.permutation(idx)
    for j in range(0, len(idx), bs):
        yield idx[j:j + bs]


def to_dev(D, b, dev):
    g = torch.from_numpy(D["grid"][b].astype(np.float32)).to(dev)
    s = torch.from_numpy(D["scal"][b].astype(np.float32)).to(dev)
    m = torch.from_numpy(np.unpackbits(D["mask"][b], axis=1)[:, :A].astype(bool)).to(dev)
    return g, s, m


def logp_of(agent, g, s, m):
    lo, _ = agent(g, s)
    return F.log_softmax(lo.masked_fill(~m, -1e9), dim=1)


@torch.no_grad()
def evaluate(student, orig, D, idx, dev, bs=4096):
    """분류별: 정답 로그 확률·확률·최빈 일치, 개입 상태의 기존 선택 확률, 원본 대비 KL"""
    acc = {}
    for b in batches(idx, bs):
        g, s, m = to_dev(D, b, dev)
        ls, lo = logp_of(student, g, s, m), logp_of(orig, g, s, m)
        a = torch.from_numpy(D["act"][b]).long().to(dev)
        base = torch.from_numpy(D["base"][b]).long().to(dev)
        lp = ls.gather(1, a[:, None])[:, 0]
        lp0 = lo.gather(1, a[:, None])[:, 0]
        top = (ls.argmax(1) == a).float()
        top0 = (lo.argmax(1) == a).float()
        pb = ls.gather(1, base[:, None])[:, 0].exp()
        pb0 = lo.gather(1, base[:, None])[:, 0].exp()
        kl = (lo.exp() * (lo - ls)).masked_fill(~m, 0).sum(1)
        cat = D["cat"][b]
        for c in np.unique(cat):
            sel = torch.from_numpy(cat == c).to(dev)
            r = acc.setdefault(int(c), np.zeros(9))
            r += np.array([sel.sum().item(), lp[sel].sum().item(), lp[sel].exp().sum().item(), top[sel].sum().item(),
                           lp0[sel].exp().sum().item(), top0[sel].sum().item(), pb[sel].sum().item(), pb0[sel].sum().item(), kl[sel].sum().item()])
    out = {}
    for c, r in acc.items():
        n = r[0]
        out[c] = {"n": int(n), "nll": -r[1] / n, "p": r[2] / n, "top1": r[3] / n, "p0": r[4] / n, "top1_0": r[5] / n,
                  "p_base": r[6] / n, "p_base0": r[7] / n, "kl": r[8] / n}
    return out


def show(tag, ev):
    rows = []
    for c in sorted(ev):
        e = ev[c]
        rows.append(f"{NAMES[c]}({e['n']}): 확률 {e['p0']:.3f}→{e['p']:.3f} · 최빈 일치 {e['top1_0']:.1%}→{e['top1']:.1%} · 손실 {e['nll']:.3f}"
                    + (f" · 기존 선택 확률 {e['p_base0']:.3f}→{e['p_base']:.3f}" if c in (1, 2) else "") + f" · KL {e['kl']:.4f}")
    print(f"  [{tag}] " + "\n  " .join(rows), flush=True)


def score(ev):
    """체크포인트 선택: 검증 교사 개입(1·2·3) 평균 로그 확률 + 유지(4) 평균 로그 확률 (두 묶음 같은 비중)"""
    n_int = sum(ev[c]["n"] for c in (1, 2, 3) if c in ev)
    lp_int = -sum(ev[c]["nll"] * ev[c]["n"] for c in (1, 2, 3) if c in ev) / n_int
    return lp_int - ev[4]["nll"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--init", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--data", default="runs/distill_s12345.npz,runs/distill_s777.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--frac", type=float, default=1.0, help="학습 게임 중 쓸 비율 (소량 확인용)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    D = load(args.data.split(","))
    val_game = D["game"] % 10 == 9
    keep = np.isin(D["cat"], TRAIN_CATS)
    tr_all = np.nonzero(~val_game & keep)[0]
    if args.frac < 1:
        games = np.unique(D["src"][tr_all] * 100000 + D["game"][tr_all])
        pick = set(rng.choice(games, int(len(games) * args.frac), replace=False).tolist())
        tr_all = tr_all[np.isin(D["src"][tr_all] * 100000 + D["game"][tr_all], list(pick))]
    va = np.nonzero(val_game)[0]
    tr_ev = rng.choice(tr_all, min(len(tr_all), 40000), replace=False)
    n_int = int(np.isin(D["cat"][tr_all], (1, 2, 3)).sum())
    print(f"학습 샘플 {len(tr_all)} (교사 개입 {n_int} · 유지 {len(tr_all) - n_int}, 개입 비율 {n_int / len(tr_all):.1%}) · "
          f"검증 샘플 {len(va)} (게임 {len(np.unique(D['src'][va] * 100000 + D['game'][va]))}판, 학습과 겹치는 게임 없음)", flush=True)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    a = ck["args"]
    orig = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    orig.load_state_dict(ck["agent"])
    orig.eval()
    for q in orig.parameters():
        q.requires_grad_(False)
    student = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    student.load_state_dict(ck["agent"])
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, eps=1e-5)
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    hist = []
    ev_v = evaluate(student, orig, D, va, dev)
    show("검증 · 학습 전", ev_v)
    best = (score(ev_v), 0)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        tot = {c: [0.0, 0] for c in TRAIN_CATS}
        for b in batches(tr_all, args.bs, rng):
            g, s, m = to_dev(D, b, dev)
            act = torch.from_numpy(D["act"][b]).long().to(dev)
            nll = -logp_of(student, g, s, m).gather(1, act[:, None])[:, 0]
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            opt.step()
            cat = D["cat"][b]
            for c in TRAIN_CATS:
                sel = cat == c
                if sel.any():
                    tot[c][0] += nll[torch.from_numpy(sel).to(dev)].sum().item()
                    tot[c][1] += int(sel.sum())
        student.eval()
        ev_t = evaluate(student, orig, D, tr_ev, dev)
        ev_v = evaluate(student, orig, D, va, dev)
        sc = score(ev_v)
        print(f"에폭 {ep} ({time.time() - t0:.0f}s) · 학습 중 손실 " + " · ".join(f"{NAMES[c]} {v[0] / max(v[1], 1):.3f}" for c, v in tot.items())
              + f" · 검증 점수 {sc:.4f}", flush=True)
        show("학습(표본)", ev_t)
        show("검증", ev_v)
        torch.save({"agent": student.state_dict(), "args": a, "rstd": ck["rstd"], "global_step": ck.get("global_step", 0),
                    "update": ck.get("update", 0), "distill": {"epoch": ep, "lr": args.lr, "init": args.init, "data": args.data}},
                   os.path.join(run_dir, f"agent_e{ep}.pt"))
        hist.append({"epoch": ep, "score": sc, "train_loss": {NAMES[c]: v[0] / max(v[1], 1) for c, v in tot.items()},
                     "val": {NAMES[c]: e for c, e in ev_v.items()}, "train_eval": {NAMES[c]: e for c, e in ev_t.items()}})
        if sc > best[0]:
            best = (sc, ep)
    print(f"최고 검증 점수 에폭 {best[1]} ({best[0]:.4f})")
    if best[1] > 0:
        import shutil
        shutil.copy(os.path.join(run_dir, f"agent_e{best[1]}.pt"), os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "best_epoch": best[1], "hist": hist}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
