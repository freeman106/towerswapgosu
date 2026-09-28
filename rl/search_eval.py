"""추론 시 탐색 평가: 결정마다 후보 행동(기본 정책이 뽑은 행동 + 확률 상위 K개 + 그 밖의 유효 행동 R개)을 적용한 게임을
가상 난수 S묶음으로 복제하고, 같은 정책으로 그날 밤이 끝날 때까지 굴려 비교한다.
  Q = (밤까지 받은 보상 / σ) + γ^경과일 · V(끝 상태)   (사망하면 V 항은 0, σ는 체크포인트의 보상 정규화 표준편차)
후보들은 같은 가상 난수 묶음과 같은 정책 샘플링 잡음(검벨)을 쓴다. 본 게임도 환경마다 한 판씩, 판·스텝별 잡음을 고정해
두므로 기본 조건과 탐색 조건은 첫 탐색 개입 전까지 똑같다(짝 비교).
사용: python rl/search_eval.py rl/runs/cb9/agent.pt [--games 32] [--k 3] [--r 2] [--s 4]"""
import argparse
import json
import time
from collections import Counter

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


DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
GROUP = {"l": "자원", "s": "자원", "i": "자원", "g": "자원", "d": "자원", "t": "타워", "b": "타워", "c": "타워", "w": "타워",
         "h": "상자", "a": "모루", "I": "빙산", "T": "TNT", "F": "요정", "H": "요정", "C": "대포 자리"}


def act_cat(cells, phase, a):
    """행동 분류 (보드 문자열 기준)"""
    if a < 168:
        c = a // 4
        x, y = c % 6 + 1, c // 6 + 1
        dx, dy = DIRS[a % 4]
        src = cells[y - 1][x - 1]
        tx, ty = x + dx, y + dy
        if ty == 0:
            return "돌 성에 넣기" if src[0] == "s" else "화살탑 포탑" if src[0] == "t" else "성 쪽 드래그"
        dst = cells[ty - 1][tx - 1] if 1 <= ty <= 7 and 1 <= tx <= 6 else "~~"
        if dst == "~~":
            return "버리기"
        if dst == "..":
            return "빈칸 이동"
        g1, g2 = sorted((GROUP.get(src[0], "기타"), GROUP.get(dst[0], "기타")))
        return f"교환 {g1}-{g2}"
    if a < 216:
        c = a - 168
        x, y = c % 6 + 1, c // 6
        if phase in (83, 43, 116, 109):
            return {83: "상인 판매", 43: "아이템 배치", 116: "TNT 폭파", 109: "요정 선택"}[phase]
        t = cells[y - 1][x - 1] if y >= 1 else ""
        return {"h": "상자 열기", "c": "대포 방향", "T": "TNT 사용", "F": "요정 사용", "H": "요정 사용"}.get(t[:1], "탭")
    if a in (216, 217):
        yes = a == 216
        return {56: "악마 " + ("수락" if yes else "거절"), 79: "상점 " + ("받기" if yes else "거절"),
                108: "휴식일 " + ("받기" if yes else "거절"), 109: "요정 취소"}.get(phase, "예" if yes else "아니오")
    if a < 223:
        return "지니"
    if a == 223:
        return "TNT 전부 폭파"
    return "밤 시작" if phase == 25 else "휴식일 끝"


def board_info(cells, turrets):
    """타워 점유율(땅 칸 중 타워 칸), 등급별 공격 타워 수"""
    land = sum(1 for row in cells for c in row if c != "~~")
    towers = sum(1 for row in cells for c in row if c[:1] in "tbcw" and c)
    tiers = [0] * 5
    for c in [c for row in cells for c in row] + list(turrets):
        if c[:1] in ("t", "b", "c"):
            tiers[min(max(int(c[1]), 1), 4)] += 1
    return towers / land, tiers


class NoSearchDayOff:
    """play()의 ctl: 휴식일(수락한 날)에는 탐색하지 않고 정책만 둔다. until > 0이면 그날 이후(until + 1일부터)도 탐색하지 않는다"""

    def __init__(self, until=0):
        self.until = until

    def before(self, env, alive):
        nosearch = np.zeros(len(alive), bool)
        for i in np.nonzero(alive)[0]:
            day = env.state(int(i))[0]
            nosearch[i] = (day % 10 == 1 and env.made(int(i))[2] == day) or (self.until > 0 and day > self.until)
        return {}, nosearch

    def after(self, env, i, a):
        pass


def play(pol, n, seed, search, k, r, s, stats, z=0.0, log=None, rec=None, frames=None, env=None, ctl=None, no_toss=False, no_open=False,
         fin_rec=None, env_kw=None):
    """log: 분석용 기록(dict, 결정·하루 단위), rec: 증류용 기록(dict of lists, 모든 스텝),
    frames: 리플레이용 기록(게임별 목록, 행동 직전 상태·보드·고른 행동·정책 상위 행동·탐색 후보),
    env: 이미 상태를 넣어 둔 VecEnv(되돌린 상태에서 이어 두기). 주면 새 게임을 시작하지 않는다,
    ctl: 행동 개입. ctl.before(env, alive) → (게임 → 강제 행동, 탐색 안 할 게임 bool 배열), ctl.after(env, i, 행동)은 행동 직전에 호출,
    ctl.choose(env, alive, logits, noise, act)가 있으면 정책 로짓·샘플링 잡음을 보고 강제 행동(dict)을 더 고른다,
    no_toss: 새로 만드는 환경에 휴식일 버리기 금지(탐색 굴리기에도 적용), no_open: 일반 상자 개봉 금지,
    fin_rec: 목록을 주면 게임별 종료 기록(pop_finished의 dict)을 모은다,
    env_kw: 새로 만드는 VecEnv에 더 넘길 인자(예: open_min_tier, emergency). frames[i]가 None인 게임은 프레임을 기록하지 않는다"""
    fresh = env is None
    if fresh:
        env = ts.VecEnv(n, seed=seed, no_toss_day_off=no_toss, no_open_normal=no_open, **(env_kw or {}))
    grid = np.zeros((n, C, H, W), np.float32)
    scal = np.zeros((n, S), np.float32)
    mask = np.zeros((n, A), bool)
    rew, done, days = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
    if fresh:
        env.reset(grid, scal, mask)
    else:
        env.observe(grid, scal, mask)
    nc = 1 + k + r
    if search:
        m = n * nc * s
        se = ts.SearchEnv(m)
        sg = np.zeros((m, C, H, W), np.float32)
        ss = np.zeros((m, S), np.float32)
        sm = np.zeros((m, A), bool)
        sa = np.zeros(m, bool)
        ret, sdays, dead, lost = np.zeros(m, np.float32), np.zeros(m, np.float32), np.zeros(m, bool), np.zeros(m, np.float32)
    alive = np.ones(n, bool)
    final = np.zeros(n, int)
    stones = np.zeros(n, int)  # 오늘 돌을 성에 넣은 횟수
    step = 0
    t_start = time.time()
    while alive.any():
        if search and step % 50 == 0:
            print(f"  [탐색] 스텝 {step} · 진행 중 {alive.sum()}판 · {time.time() - t_start:.0f}s", flush=True)
        logits, _ = pol(grid, scal, mask)
        noise = np.random.default_rng([seed, step]).gumbel(size=(n, A))
        act = (logits + noise).argmax(1)
        base_act = act.copy()
        forced, nosearch = ctl.before(env, alive) if ctl else ({}, np.zeros(n, bool))
        if ctl is not None and hasattr(ctl, "choose"):
            forced.update(ctl.choose(env, alive, logits, noise, act))
        for i in forced:
            nosearch[i] = True
        info = {}  # 게임 → (후보, 후보별 평균 Q, 기본 대비 차이 평균, 표준오차)
        if search:
            # 후보: 기본 행동, 확률 상위 k개, 그 밖의 유효 행동 r개
            # 무작위 후보는 게임별 난수 행(판·스텝마다 고정)으로 고른다: 다른 게임의 상태·개입이 이 게임의 후보를 바꾸지 않는다
            keys = np.random.default_rng([seed, step, 11]).random((n, A))
            idx, first, which = [], [], []
            for i in np.nonzero(alive & (mask.sum(1) >= 2) & ~nosearch)[0]:
                valid = np.nonzero(mask[i])[0]
                top = valid[np.argsort(-logits[i, valid])[:k]]
                cand = [int(act[i])] + [int(a) for a in top if a != act[i]]
                rest = np.array([a for a in valid if a not in cand], np.int64)
                cand += [int(a) for a in rest[np.argsort(keys[i, rest])[:r]]]
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
                    info[i] = (cands, qs.mean(1), np.r_[0.0, mean], np.r_[0.0, sem])
                    if ok.any():
                        j = int(np.argmax(np.where(ok, mean, -np.inf)))
                        c = int(cands[1 + j])
                        stats["changed"] += 1
                        stats["rank"][min(int((logits[i] > logits[i, c]).sum()), 4)] += 1
                        stats["gain"].append(float(mean[j]))
                        act[i] = c
        for i, a in forced.items():
            act[i] = a
        if log is not None:
            for i in np.nonzero(alive)[0]:
                day, hearts, swaps, phase, _ = env.state(int(i))
                cells, tur = env.board(int(i))
                cat = act_cat(cells, phase, int(act[i]))
                stones[i] += cat == "돌 성에 넣기"
                if cat == "밤 시작" or cat == "휴식일 끝":
                    occ, tiers = board_info(cells, tur)
                    log["dusk"].append({"game": int(i), "day": day, "hearts": hearts, "occ": occ, "tiers": tiers, "stones": int(stones[i])})
                    stones[i] = 0
                if i in info:
                    cands, qm, dm, sd = info[i]
                    occ, _ = board_info(cells, tur)
                    ch = int(act[i]) != int(base_act[i])
                    log["dec"].append({"game": int(i), "day": day, "hearts": hearts, "swaps": swaps, "phase": phase, "occ": occ,
                                       "base": act_cat(cells, phase, int(base_act[i])), "chosen": cat, "changed": ch,
                                       "gain": float(dm[list(cands).index(act[i])]) if ch else 0.0,
                                       "rank": int((logits[i] > logits[i, act[i]]).sum())})
        if frames is not None:
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            for i in np.nonzero(alive)[0]:
                if frames[i] is None:
                    continue
                day, hearts, swaps, phase, score = env.state(int(i))
                cells, tur = env.board(int(i))
                top = np.argsort(-probs[i])[:3]
                f = {"d": day, "h": hearts, "s": swaps, "p": phase, "sc": score, "b": [c for row in cells for c in row], "t": tur,
                     "a": int(act[i]), "pa": round(float(probs[i, act[i]]), 3),
                     "top": [[int(x), round(float(probs[i, x]), 3)] for x in top if probs[i, x] > 0.005]}
                if i in info:
                    cands, qm, dm, sd = info[i]
                    f["base"] = int(base_act[i])
                    f["cand"] = [[int(c), round(float(m_), 4), round(float(e_), 4)] for c, m_, e_ in zip(cands, dm, sd)]
                frames[i].append(f)
        if rec is not None:
            for i in np.nonzero(alive)[0]:
                rec["game"].append(int(i))
                rec["grid"].append(grid[i].astype(np.float16))
                rec["scal"].append(scal[i].astype(np.float16))
                rec["mask"].append(np.packbits(mask[i]))
                rec["act"].append(int(act[i]))
                cands, qm, dm, sd = info.get(i, (np.array([act[i]]), np.zeros(1), np.zeros(1), np.zeros(1)))
                pad = lambda x, v: np.r_[x, np.full(nc - len(x), v)][:nc]
                rec["cands"].append(pad(np.asarray(cands, np.int64), -1).astype(np.int16))
                rec["qmean"].append(pad(qm, np.nan).astype(np.float32))
                rec["dmean"].append(pad(dm, np.nan).astype(np.float32))
                rec["dsem"].append(pad(sd, np.nan).astype(np.float32))
                rec["searched"].append(i in info)
        if ctl:
            for i in np.nonzero(alive)[0]:
                ctl.after(env, i, int(act[i]))
        env.step(act, grid, scal, mask, rew, done, days)
        fins = {f["env"]: f for f in env.pop_finished()}
        if rec is not None:
            for i in np.nonzero(alive)[0]:
                rec["rew"].append(float(rew[i]))
                rec["days"].append(float(days[i]))
                rec["done"].append(bool(done[i]))
        if log is not None:
            for i in np.nonzero((days > 0) & alive & ~done)[0]:
                day, hearts, _, _, _ = env.state(int(i))
                log["morning"].append({"game": int(i), "day": day, "hearts": hearts})
        for i, f in fins.items():
            if alive[i]:
                final[i] = f["day"]
                if fin_rec is not None:
                    fin_rec.append(f)
                alive[i] = False
        step += 1
    return final


def summary(name, fin, secs):
    r30 = fin >= 30
    score = np.mean([d + 1000 * min(5, (d - 1) // 10) for d in fin])
    return (f"{name}: day {fin.mean():.2f} · 점수 {score:.0f} · 30일 도달 {r30.mean():.1%} · "
            f"도달 후 통과 {(fin > 30).sum() / max(r30.sum(), 1):.1%} · 32·35·40일 도달 {np.mean(fin >= 32):.1%}/{np.mean(fin >= 35):.1%}/{np.mean(fin >= 40):.1%} · 실행 {secs:.0f}s\n"
            f"  사망일 분포(25일+): {dict(sorted(Counter(int(d) for d in fin if d >= 25).items()))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--k", type=int, default=3, help="확률 상위 후보 수")
    p.add_argument("--r", type=int, default=2, help="그 밖의 무작위 유효 행동 후보 수")
    p.add_argument("--s", type=int, default=4, help="가상 난수 묶음 수")
    p.add_argument("--z", type=float, default=2.0, help="기본 행동 대비 짝 차이 평균이 z·표준오차보다 클 때만 바꾼다 (0이면 평균만 비교)")
    p.add_argument("--log", default="", help="분석용 기록(JSON) 경로: 결정·하루 단위 (두 조건 모두)")
    p.add_argument("--record", default="", help="증류용 기록(npz) 경로: 탐색 조건의 모든 스텝 (관측, 행동, 후보별 Q, 보상)")
    p.add_argument("--no-toss-day-off", action="store_true", help="휴식일 버리기 금지 (두 조건 모두)")
    p.add_argument("--no-search-day-off", action="store_true", help="휴식일에는 탐색하지 않는다")
    p.add_argument("--no-open-normal", action="store_true", help="일반(1등급) 상자 개봉 금지 (두 조건 모두)")
    p.add_argument("--search-until", type=int, default=0, help="이 날까지만 탐색한다(다음 날부터 정책만). 0이면 끝까지")
    p.add_argument("--out", default="", help="게임별 최종 일차(JSON) 저장 경로: 짝 비교용")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    pol = Policy(args.ckpt, torch.device(args.device))
    print(f"σ {pol.sigma:.3f} · γ {pol.gamma} · 후보 기본+상위 {args.k}+무작위 {args.r} · 난수 묶음 {args.s} · z {args.z}", flush=True)

    logs = {c: {"dec": [], "dusk": [], "morning": []} for c in ("base", "search")} if args.log else {"base": None, "search": None}
    rec = {k_: [] for k_ in ("game", "grid", "scal", "mask", "act", "cands", "qmean", "dmean", "dsem", "searched", "rew", "days", "done")} if args.record else None
    t0 = time.time()
    base = play(pol, args.games, args.seed, False, 0, 0, 0, None, log=logs["base"], no_toss=args.no_toss_day_off, no_open=args.no_open_normal)
    tb = time.time() - t0
    stats = {"searched": 0, "changed": 0, "rank": [0] * 5, "gain": [], "rollout_steps": 0}
    t0 = time.time()
    srch = play(pol, args.games, args.seed, True, args.k, args.r, args.s, stats, args.z, log=logs["search"], rec=rec,
                ctl=NoSearchDayOff(args.search_until) if args.no_search_day_off or args.search_until else None,
                no_toss=args.no_toss_day_off, no_open=args.no_open_normal)
    ts_ = time.time() - t0
    if args.log:
        with open(args.log, "w") as f:
            json.dump({"final": {"base": base.tolist(), "search": srch.tolist()}, **logs}, f, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"base": base.tolist(), "search": srch.tolist()}, f)
    if rec is not None:
        np.savez_compressed(args.record, final=srch, sigma=pol.sigma, gamma=pol.gamma, z=args.z,
                            **{k_: np.array(v) for k_, v in rec.items()})
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
