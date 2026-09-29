"""작은 고정 집합 암기 검사 (진단 전용, 게임 평가에 쓰지 않는다). 원본 cn7_nf 20M에서 별도 진단 학생을 시작해
교사 한 수·교사 준비·실제 완성에서 각 N개(학습 게임에서, 원본 정답 로그 확률의 분위를 고르게: 매우 낮은 표본 포함)를 고정하고,
환경 마스크 안에서 이 표본만 교차 엔트로피로 반복 학습한다(전체 배치, 유지 샘플·KL 없음).
기록(LOG마다): 분류별 NLL 평균, 정답 확률 중앙값, 정답 순위 중앙값(0 = 최빈), 최빈 일치율. 전체 최빈 일치 ≥ 목표면 조기 종료.
함께 기록: gradient clipping 작동 비율·클립 전 norm, 층별 gradient norm·실제 갱신량, 중복 관측의 상충 정답, 정답의 마스크 유효성.
사용: python rl/distill_probe.py [--lr 3e-5 --max-updates 1000]"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from distill_student import NAMES, load, to_dev
from ppo import A, Agent

CATS = (1, 2, 3)


def stats(net, g, s, m, a, cat):
    lo, _ = net(g, s)
    lo = lo.masked_fill(~m, -1e9)
    lp = F.log_softmax(lo, 1)
    lpa = lp.gather(1, a[:, None])[:, 0]
    rank = (lo > lo.gather(1, a[:, None])).sum(1)
    top = (lo.argmax(1) == a)
    ent = -(lp.exp() * lp.masked_fill(~m, 0)).sum(1)
    out = {}
    for c in CATS + (0,):
        sel = torch.ones_like(top) if c == 0 else torch.from_numpy(cat == c).to(lo.device)
        out["전체" if c == 0 else NAMES[c]] = {
            "NLL": float(-lpa[sel].mean()), "확률 중앙": float(lpa[sel].exp().median()), "순위 중앙": float(rank[sel].float().median()),
            "최빈 일치": float(top[sel].float().mean()), "엔트로피": float(ent[sel].mean())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--data", default="runs/distill_s12345.npz,runs/distill_s777.npz")
    p.add_argument("--n-per", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--clip", type=float, default=0.5, help="gradient clipping norm (기존 학습과 같은 0.5)")
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/distill_probe")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    D = load(args.data.split(","))
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    a_ = ck["args"]
    orig = Agent(a_["channels"], a_["blocks"], a_["hidden"]).to(dev)
    orig.load_state_dict(ck["agent"])
    orig.eval()
    # 분류별 후보: 학습 게임, 정답이 마스크에서 유효. 원본 로그 확률 분위를 고르게 N개
    train = D["game"] % 10 != 9
    picks = []
    for c in CATS:
        cand = np.nonzero(train & (D["cat"] == c))[0]
        with torch.no_grad():
            lps = []
            for j in range(0, len(cand), 4096):
                b = cand[j:j + 4096]
                g, s, m = to_dev(D, b, dev)
                lo, _ = orig(g, s)
                lps.append(F.log_softmax(lo.masked_fill(~m, -1e9), 1).gather(1, torch.from_numpy(D["act"][b]).long().to(dev)[:, None])[:, 0].cpu().numpy())
        lp = np.concatenate(lps)
        order = cand[np.argsort(lp)]
        sel = order[np.linspace(0, len(order) - 1, args.n_per).round().astype(int)]
        picks.append(sel)
        print(f"{NAMES[c]}: 후보 {len(cand)} · 고른 표본 원본 log p 범위 {np.sort(lp)[0]:.1f} ~ {np.sort(lp)[-1]:.1f}")
    idx = np.concatenate(picks)
    g, s, m = to_dev(D, idx, dev)
    a = torch.from_numpy(D["act"][idx]).long().to(dev)
    cat = D["cat"][idx]
    valid = m.gather(1, a[:, None])[:, 0]
    print(f"정답 행동이 환경 마스크에서 유효: {int(valid.sum())}/{len(idx)}")
    assert bool(valid.all())
    keys = [hashlib.sha1(D["grid"][i].tobytes() + D["scal"][i].tobytes() + D["mask"][i].tobytes()).hexdigest() for i in idx]
    dup = {}
    for k, i in zip(keys, idx):
        dup.setdefault(k, set()).add(int(D["act"][i]))
    print(f"같은 관측 묶음 {len(dup)}개 (표본 {len(idx)}) · 같은 관측에 다른 정답 {sum(1 for v in dup.values() if len(v) > 1)}묶음")
    student = Agent(a_["channels"], a_["blocks"], a_["hidden"]).to(dev)
    student.load_state_dict(ck["agent"])
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, eps=1e-5)
    p0 = {n: q.detach().clone() for n, q in student.named_parameters()}
    log, clip_norms, layer = [], [], {}
    with torch.no_grad():
        st = stats(student, g, s, m, a, cat)
    log.append({"갱신": 0, **st})
    print(f"갱신 0: " + " · ".join(f"{k} NLL {v['NLL']:.2f}·확률 중앙 {v['확률 중앙']:.2e}·순위 {v['순위 중앙']:.0f}·일치 {v['최빈 일치']:.0%}" for k, v in st.items()), flush=True)
    u = 0
    for u in range(1, args.max_updates + 1):
        student.train()
        lo, _ = student(g, s)
        nll = -F.log_softmax(lo.masked_fill(~m, -1e9), 1).gather(1, a[:, None])[:, 0]
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        if u in (1, 10, 100) or u % 250 == 0:
            layer[u] = {n: (float(q.grad.norm()) if q.grad is not None else 0.0) for n, q in student.named_parameters()}
        norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip))
        clip_norms.append(norm)
        opt.step()
        if u % args.log_every == 0 or u == args.max_updates:
            student.eval()
            with torch.no_grad():
                st = stats(student, g, s, m, a, cat)
            log.append({"갱신": u, **st, "클립 전 norm 중앙(최근)": float(np.median(clip_norms[-args.log_every:]))})
            print(f"갱신 {u}: " + " · ".join(f"{k} NLL {v['NLL']:.2f}·확률 중앙 {v['확률 중앙']:.2e}·순위 {v['순위 중앙']:.0f}·일치 {v['최빈 일치']:.0%}" for k, v in st.items())
                  + f" · 클립 전 norm 중앙 {np.median(clip_norms[-args.log_every:]):.2f}", flush=True)
            if st["전체"]["최빈 일치"] >= args.target:
                print(f"전체 최빈 일치 {st['전체']['최빈 일치']:.0%} ≥ {args.target:.0%}: 조기 종료")
                break
    cn = np.array(clip_norms)
    moved = {n: float((q.detach() - p0[n]).norm() / (p0[n].norm() + 1e-12)) for n, q in student.named_parameters()}
    print(f"gradient clipping: 작동 {np.mean(cn > args.clip):.0%} (클립 전 norm 중앙 {np.median(cn):.2f}, 최대 {cn.max():.2f}, 기준 {args.clip})")
    print("층별 상대 갱신량(‖Δθ‖/‖θ‖): " + " · ".join(f"{n} {v:.2e}" for n, v in moved.items()))
    os.makedirs(args.out, exist_ok=True)
    torch.save({"agent": student.state_dict(), "args": a_, "rstd": ck["rstd"], "probe": vars(args)}, os.path.join(args.out, "probe.pt"))
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump({"args": vars(args), "log": log, "clip_norms": clip_norms, "layer_grad": layer, "moved": moved, "idx": idx.tolist(),
                   "updates": u}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
