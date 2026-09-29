"""기존 운영 보존 실험: 원본 cn7_nf 20M에서 조건마다 새로 시작한 학생을 같은 학습·검증 분리, 같은 데이터 순서·배치, 같은 학습률·clipping,
같은 갱신 수로 학습한다(암기 검사 모델이나 이전 학생에서 이어 학습하지 않는다).
  A: 교차 엔트로피만 — 교사 한 수(1)·준비(2)·실제 완성(3)·유지(4) 샘플에서 실제로 둔 행동의 NLL, 배치 안 해당 샘플 평균.
  B: A와 같은 손실 + β · KL(원본 정책 ‖ 학생 정책) — 유지(4)·그 밖(0)·재확인 실패 뒤(5) 샘플에서, 배치 안 해당 샘플 평균.
     교사 한 수·준비·실제 완성 상태에는 KL을 걸지 않는다.
손실 = (교차 엔트로피 샘플 평균) + β · (KL 샘플 평균). 표본 상한·낮은 확률 표본 제외 없음. 배치는 학습 샘플 전체(분류 0~5)를 한 번 섞은 순서로 자른다.
기록(갱신 LOG마다, 검증 게임): 교사 분류별 NLL·확률 중앙값·순위 중앙값·최빈 일치, 유지 상태 KL(원본‖학생)·원본 최빈 선택 유지율·원래 둔 행동 확률,
  분류별 정책 엔트로피, 그 밖·재확인 실패 뒤 KL. 학습 게임 표본에도 같은 지표. 체크포인트는 SAVE마다.
사용: python rl/distill_ab.py --run distill_A --beta 0 / --run distill_B1 --beta 1"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from distill_student import NAMES, load, to_dev
from ppo import A, Agent

CE_CATS = (1, 2, 3, 4)
KL_CATS = (0, 4, 5)


def net_from(ck, dev):
    a = ck["args"]
    m = Agent(a["channels"], a["blocks"], a["hidden"]).to(dev)
    m.load_state_dict(ck["agent"])
    return m


@torch.no_grad()
def evaluate(student, orig, D, idx, dev, bs=4096):
    acc = {}
    for j in range(0, len(idx), bs):
        b = idx[j:j + bs]
        g, s, m = to_dev(D, b, dev)
        lo_s, _ = student(g, s)
        lo_o, _ = orig(g, s)
        lo_s, lo_o = lo_s.masked_fill(~m, -1e9), lo_o.masked_fill(~m, -1e9)
        ls, lo = F.log_softmax(lo_s, 1), F.log_softmax(lo_o, 1)
        a = torch.from_numpy(D["act"][b]).long().to(dev)
        lp = ls.gather(1, a[:, None])[:, 0]
        rank = (lo_s > lo_s.gather(1, a[:, None])).sum(1).float()
        top = (lo_s.argmax(1) == a).float()
        keep = (lo_s.argmax(1) == lo_o.argmax(1)).float()
        kl = (lo.exp() * (lo - ls)).masked_fill(~m, 0).sum(1)
        ent = -(ls.exp() * ls.masked_fill(~m, 0)).sum(1)
        ent0 = -(lo.exp() * lo.masked_fill(~m, 0)).sum(1)
        cat = D["cat"][b]
        for c in np.unique(cat):
            sel = torch.from_numpy(cat == c).to(dev)
            r = acc.setdefault(int(c), {k: [] for k in ("lp", "rank", "top", "keep", "kl", "ent", "ent0")})
            for k, v in (("lp", lp), ("rank", rank), ("top", top), ("keep", keep), ("kl", kl), ("ent", ent), ("ent0", ent0)):
                r[k].append(v[sel].cpu().numpy())
    out = {}
    for c, r in acc.items():
        r = {k: np.concatenate(v) for k, v in r.items()}
        out[NAMES[c]] = {"n": int(len(r["lp"])), "NLL": float(-r["lp"].mean()), "확률 중앙": float(np.median(np.exp(r["lp"]))),
                         "순위 중앙": float(np.median(r["rank"])), "최빈 일치": float(r["top"].mean()), "원본 최빈 유지": float(r["keep"].mean()),
                         "KL": float(r["kl"].mean()), "엔트로피": float(r["ent"].mean()), "원본 엔트로피": float(r["ent0"].mean())}
    return out


def line(tag, ev):
    t = " · ".join(f"{k} NLL {ev[k]['NLL']:.2f}·확률 중앙 {ev[k]['확률 중앙']:.2e}·순위 {ev[k]['순위 중앙']:.0f}·일치 {ev[k]['최빈 일치']:.1%}·엔트로피 {ev[k]['엔트로피']:.2f}"
                   for k in ("교사 한 수", "교사 준비", "실제 완성") if k in ev)
    k = ev["유지"]
    t2 = (f"유지 KL {k['KL']:.4f}·원본 최빈 유지 {k['원본 최빈 유지']:.1%}·둔 행동 확률 중앙 {k['확률 중앙']:.3f}·엔트로피 {k['원본 엔트로피']:.2f}→{k['엔트로피']:.2f}"
          + "".join(f" · {n} KL {ev[n]['KL']:.4f}·유지 {ev[n]['원본 최빈 유지']:.1%}" for n in ("그 밖", "재확인 실패 뒤") if n in ev))
    print(f"  [{tag}] {t}\n  [{tag}] {t2}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--init", default="runs/cn7_nf/agent_20M.pt")
    p.add_argument("--data", default="runs/distill_s12345.npz,runs/distill_s777.npz")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    D = load(args.data.split(","))
    val = D["game"] % 10 == 9
    tr = np.nonzero(~val)[0]
    order = np.random.default_rng([args.seed, 1]).permutation(tr)  # 조건과 무관하게 같은 순서
    rng = np.random.default_rng([args.seed, 2])
    va = np.nonzero(val)[0]
    va_ev = np.concatenate([va[np.isin(D["cat"][va], (0, 1, 2, 3, 5))], rng.choice(va[D["cat"][va] == 4], 6000, replace=False)])
    tr_ev = np.concatenate([rng.choice(tr[D["cat"][tr] == c], min(3000, int((D["cat"][tr] == c).sum())), replace=False) for c in (1, 2, 3, 4)])
    cnt = {NAMES[c]: int((D["cat"][tr] == c).sum()) for c in range(6)}
    n_ce = sum(cnt[NAMES[c]] for c in CE_CATS)
    n_kl = sum(cnt[NAMES[c]] for c in KL_CATS)
    print(f"학습 샘플 {len(tr)} ({cnt}) · 교차 엔트로피 대상 {n_ce} (교사 {n_ce - cnt['유지']} = {1 - cnt['유지'] / n_ce:.1%}) · "
          f"KL 대상 {n_kl if args.beta else 0} · β {args.beta} · 배치 {args.bs} · 갱신 {args.updates} (학습 샘플 {args.updates * args.bs / len(tr):.2f}회 분량)", flush=True)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    orig = net_from(ck, dev)
    orig.eval()
    for q in orig.parameters():
        q.requires_grad_(False)
    student = net_from(ck, dev)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, eps=1e-5)
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    hist = [{"갱신": 0, "검증": evaluate(student, orig, D, va_ev, dev), "학습": evaluate(student, orig, D, tr_ev, dev)}]
    line("검증 0", hist[0]["검증"])
    ce_is = np.isin(D["cat"], CE_CATS)
    kl_is = np.isin(D["cat"], KL_CATS)
    t0, pos, clipn = time.time(), 0, []
    run_loss = {"ce": [], "kl": []}
    for u in range(1, args.updates + 1):
        if pos + args.bs > len(order):
            pos = 0
        b = order[pos:pos + args.bs]
        pos += args.bs
        student.train()
        g, s, m = to_dev(D, b, dev)
        lo = student(g, s)[0].masked_fill(~m, -1e9)
        ls = F.log_softmax(lo, 1)
        a = torch.from_numpy(D["act"][b]).long().to(dev)
        ce_sel = torch.from_numpy(ce_is[b]).to(dev)
        ce = -ls.gather(1, a[:, None])[:, 0][ce_sel].mean()
        loss = ce
        klv = torch.zeros((), device=dev)
        if args.beta > 0:
            kl_sel = torch.from_numpy(kl_is[b]).to(dev)
            if kl_sel.any():
                with torch.no_grad():
                    lo0 = F.log_softmax(orig(g[kl_sel], s[kl_sel])[0].masked_fill(~m[kl_sel], -1e9), 1)
                klv = (lo0.exp() * (lo0 - ls[kl_sel])).masked_fill(~m[kl_sel], 0).sum(1).mean()
                loss = ce + args.beta * klv
        opt.zero_grad()
        loss.backward()
        clipn.append(float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)))
        opt.step()
        run_loss["ce"].append(float(ce))
        run_loss["kl"].append(float(klv))
        if u % args.log_every == 0:
            student.eval()
            ev_v, ev_t = evaluate(student, orig, D, va_ev, dev), evaluate(student, orig, D, tr_ev, dev)
            hist.append({"갱신": u, "검증": ev_v, "학습": ev_t, "손실 CE": float(np.mean(run_loss["ce"][-args.log_every:])),
                         "손실 KL": float(np.mean(run_loss["kl"][-args.log_every:])), "클립 전 norm 중앙": float(np.median(clipn[-args.log_every:]))})
            print(f"갱신 {u} ({time.time() - t0:.0f}s) · 학습 손실 CE {hist[-1]['손실 CE']:.3f} · KL {hist[-1]['손실 KL']:.4f} · 클립 전 norm 중앙 {hist[-1]['클립 전 norm 중앙']:.2f}", flush=True)
            line("검증", ev_v)
            if u % 250 == 0:
                line("학습 표본", ev_t)
        if u % args.save_every == 0:
            torch.save({"agent": student.state_dict(), "args": ck["args"], "rstd": ck["rstd"], "global_step": ck.get("global_step", 0),
                        "distill": {"updates": u, "beta": args.beta, "lr": args.lr, "init": args.init}}, os.path.join(run_dir, f"agent_u{u}.pt"))
    cn = np.array(clipn)
    print(f"clipping 작동 {np.mean(cn > args.clip):.0%} · 클립 전 norm 중앙 {np.median(cn):.2f} · 최대 {cn.max():.2f}")
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "counts": cnt, "hist": hist}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
