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
    p.add_argument("--chest-bonus", default="", help="등급 1..4 상자를 게임에서 처음 만들 때의 보상, 예: 0.1,0.5,2,5")
    p.add_argument("--econ-shaping", type=float, default=0.0,
                   help="퍼텐셜 기반 경제 조형의 스왑 1개당 가중치 w: Φ = w·(남은 스왑 + 보드 상자의 개봉 스왑), 0이면 끔")
    p.add_argument("--no-toss-day-off", type=int, default=0, help="1이면 휴식일 버리기 금지 (행동 마스크 제약)")
    p.add_argument("--no-toss", type=int, default=0, help="1이면 모든 날 버리기 금지 (엔진 버리기 판정, 행동 마스크 제약)")
    p.add_argument("--no-cannon-flip", type=int, default=0, help="1이면 낮의 대포 방향 전환 금지 (행동 마스크 제약)")
    p.add_argument("--no-open-normal", type=int, default=0, help="1이면 일반(1등급) 상자 개봉 금지, 동상자부터 연다 (--open-min-tier 2와 같음)")
    p.add_argument("--open-min-tier", type=int, default=1, help="평소 열 수 있는 최소 상자 등급 (행동 마스크 제약, 1이면 제한 없음)")
    p.add_argument("--emergency", default="", help="비상 예외 '등급,스왑 상한,하트 상한' (예: 2,0,5): 그 등급 상자는 스왑·하트가 상한 이하이고 "
                                                   "지금 열 수 있는 최소 등급 이상 상자가 없을 때만 열 수 있다. 비우면 없음")
    p.add_argument("--merge-rule", default="",
                   help="합성 우선 규칙 '남은 스왑 하한,하루 최대 횟수' (예: 20,6): 은상자 이상을 연 날 기본→동 무기 합성 드래그가 있으면 "
                        "행동을 그 드래그로 제한한다 (행동 마스크 제약). 비우면 없음")
    p.add_argument("--reinvest-bonus", default="",
                   help="재투자 보상 '은,금' (예: 3,16.3): 합성으로 만든 은·금상자를 열 때 더한다(연습 배치·상점·악마 거래 상자 제외). 비우면 없음")
    p.add_argument("--silver-starts", type=int, default=0,
                   help="은상자 연습(난이도 6) 시작 상태 수: --pool-ckpt 정책이 실제로 둔 게임의 낮 상태에서 1~6행 자원 하나를 은상자로 바꾼다. "
                        "--start-frac 비율로 새 게임을 이 풀에서 시작한다(연습 게임도 끝까지 둔다)")
    p.add_argument("--stage-starts", default="",
                   help="동상자 연습 배치 (bronze3: 교환 한 번이면 은상자가 되는 동상자 3개, 난이도 7 / bronze2: 동상자 2개, 난이도 8). "
                        "--silver-starts와 같은 수집 상태에 놓아 은상자 연습과 같은 수만큼 풀에 더한다(--start-frac는 두 풀 합)")
    p.add_argument("--pool-ckpt", default="", help="연습 상태를 모을 정책 (비우면 --init)")
    p.add_argument("--pool-games", type=int, default=512)
    p.add_argument("--pool-seed", type=int, default=99)
    p.add_argument("--pool-open-min-tier", type=int, default=2, help="연습 상태를 모을 때의 개봉 규칙 최소 등급")
    p.add_argument("--start-level", default="", help="연습 시작 상태의 난이도 목록, 예: 1,2 (1: 교환 한 번이면 상자 합성, 2: 두 번, "
                                                        "3: 같은 등급 상자 3개, 4: 2개). 비우면 없음")
    p.add_argument("--start-pool", type=int, default=4000, help="연습 시작 상태 수 (난이도마다 나눈다)")
    p.add_argument("--start-frac", type=float, default=0.2, help="새 게임을 연습 상태에서 시작할 확률")
    p.add_argument("--aux-coef", type=float, default=0.0,
                   help="합성 보조 손실 계수: 합성이 유리하다고 확인된 연습 시작 상태에서 -log Σ_{a∈M(s)} π(a|s)."
                        " 미니배치 크기로 나누므로 대상 샘플 하나가 어드밴티지 1인 정책 경사 샘플 × 계수의 무게를 갖는다")
    p.add_argument("--aux-confirm", type=int, default=4, help="합성 유불리 확인에 쓰는 밤 시뮬레이션 시드 쌍 수")
    p.add_argument("--aux-levels", default="1", help="보조 손실을 줄 연습 난이도 목록 (1: 합성 행동, 2: 준비 이동 → 합성 행동)")
    p.add_argument("--aux-target", type=float, default=0.3,
                   help="보조 손실 난이도 중 가장 높은 난이도의 연습 성공률(success_path)이 이 값을 넘으면 계수를 줄이기 시작")
    p.add_argument("--aux-decay", type=int, default=100, help="계수를 0으로 줄이는 데 걸리는 업데이트 수")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=50, help="업데이트 단위")
    p.add_argument("--keep-every", type=int, default=0, help="이 업데이트마다 체크포인트 사본(agent_<스텝>M.pt)을 남긴다. 0이면 끔")
    p.add_argument("--resume", default="", help="이어서 학습할 체크포인트")
    p.add_argument("--init", default="", help="가중치만 불러올 체크포인트 (예: bc.py 결과). 네트워크 크기도 따른다")
    p.add_argument("--value-warmup", type=int, default=0, help="처음 이만큼의 업데이트는 가치 함수만 학습한다")
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
        bm, bv, bc = float(x.mean()), float(x.var()), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot
        self.ret[done] = 0.0

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


SUMMARY_KEYS = ["episodes", "score", "day", "day_max", "bosses", "r_survival", "r_boss", "r_chest", "r_econ",
                "made1", "made2", "made3", "made4", "opened1", "opened2", "opened3", "opened4",
                "chest_ge2", "chest_ge3", "chest_ge4", "hold1_steps", "hold1_days", "max_held1",
                "merge_opps", "merge_take", "hold2_frac", "hold3_frac", "games_hold2", "games_hold3", "emergency", "merge_rule",
                "r_reinvest", "reinvest3", "reinvest4"]
SILVER_LEVEL = 6  # 은상자 연습 시작의 난이도 번호
STAGE_LEVEL = {"bronze3": 7, "bronze2": 8}  # 동상자 연습 시작의 난이도 번호
PRACTICE_STATS = ("episodes", "success_path", "success_day", "ready_no_merge", "merge_take", "made2")
PRACTICE_KEYS = [f"p{lv}_{k}" for lv in (1, 2, 3, 4) for k in PRACTICE_STATS]


def episode_summary(fin):
    """끝난 게임 기록(VecEnv.pop_finished의 dict 목록)의 게임당 평균.
    보상 성분 합계(생존·보스·상자 보너스)와 등급 1..4 상자 생성·개봉 수, 동·은·금 도달 비율."""
    if not fin:
        return None
    col = lambda k: np.array([f[k] for f in fin], dtype=np.float64)
    made, opened, mc = np.array([f["made"] for f in fin]), np.array([f["opened"] for f in fin]), col("max_chest")
    s = {"episodes": len(fin), "score": col("score").mean(), "day": col("day").mean(), "day_max": col("day").max(),
         "bosses": (col("score") // 1000).mean(), "r_survival": col("r_survival").mean(), "r_boss": col("r_boss").mean(),
         "r_chest": col("r_chest").mean(), "r_econ": col("r_econ").mean()}
    for t in range(4):
        s[f"made{t + 1}"], s[f"opened{t + 1}"] = made[:, t].mean(), opened[:, t].mean()
    for k in (2, 3, 4):
        s[f"chest_ge{k}"] = (mc >= k).mean()
    n_open1 = opened[:, 0].sum()
    s["hold1_steps"] = sum(f["hold_steps"][0] for f in fin) / n_open1 if n_open1 else float("nan")
    s["hold1_days"] = sum(f["hold_days"][0] for f in fin) / n_open1 if n_open1 else float("nan")
    s["max_held1"] = col("max_normal_held").mean()
    opps = col("merge_opps")
    s["merge_opps"] = opps.mean()
    s["merge_take"] = col("merge_taken").sum() / opps.sum() if opps.sum() else float("nan")
    # 같은 등급 상자를 2개·3개 이상 동시에 가진 행동 비율과, 그런 순간이 있었던 게임 비율
    steps = col("steps").sum()
    s["hold2_frac"], s["hold3_frac"] = col("steps_hold2").sum() / steps, col("steps_hold3").sum() / steps
    s["games_hold2"], s["games_hold3"] = (col("max_same_held") >= 2).mean(), (col("max_same_held") >= 3).mean()
    s["emergency"] = col("emergency_opens").mean()
    s["merge_rule"] = col("merge_rule_uses").mean()
    s["r_reinvest"] = col("r_reinvest").mean()
    reinv = np.array([f["reinvest_opens"] for f in fin])
    s["reinvest3"], s["reinvest4"] = reinv[:, 2].mean(), reinv[:, 3].mean()
    return s


def practice_summary(fin, level=None):
    """연습 시작 게임(난이도 level, None이면 전부)의 요약.
    success_path: 난이도만큼의 행동 안에 합성 완료(1단계: 첫 행동, 2단계: 준비 이동 + 합성),
    success_day: 시작한 날 안에 합성 완료, ready_no_merge: 교환 한 번 합성 기회가 있었는데 한 번도 합성하지 않은 게임,
    merge_take: 합성 기회에서 합성한 비율, made2: 동상자를 만든 게임"""
    p = [f for f in fin if f["practice"] and (level is None or f["practice_level"] == level)]
    if not p:
        return None
    opps = sum(f["merge_opps"] for f in p)
    fm = np.array([f["first_merge_step"] for f in p])
    lv = np.array([f["practice_level"] for f in p])
    return {"episodes": len(p), "merge_opps": opps / len(p),
            "success_path": float(np.mean((fm >= 1) & (fm <= lv))),
            "success_day": float(np.mean([f["first_merge_day"] == 0 for f in p])),
            "ready_no_merge": float(np.mean([f["merge_opps"] > 0 and f["merge_taken"] == 0 for f in p])),
            "merge_take": sum(f["merge_taken"] for f in p) / opps if opps else float("nan"),
            "made2": float(np.mean([f["made"][1] > 0 for f in p]))}


def practice_row(fin):
    """CSV용: 난이도 1..4별 PRACTICE_STATS"""
    row = []
    for lv in (1, 2, 3, 4):
        ps = practice_summary(fin, lv)
        row += [ps[k] if ps else "" for k in PRACTICE_STATS]
    return row


def format_summary(s):
    if s is None:
        return "끝난 게임 없음"
    return (f"게임 {s['episodes']} 점수 {s['score']:.0f} day {s['day']:.2f}(최대 {s['day_max']:.0f}) | "
            f"보상 생존 {s['r_survival']:.3f} 보스 {s['r_boss']:.2f} 상자 {s['r_chest']:.2f} 경제 {s['r_econ']:.2f} | "
            f"상자 생성 {s['made1']:.2f}/{s['made2']:.2f}/{s['made3']:.2f}/{s['made4']:.2f} "
            f"개봉 {s['opened1']:.2f}/{s['opened2']:.2f}/{s['opened3']:.2f}/{s['opened4']:.2f} (비상 {s['emergency']:.2f}) | "
            f"일반 보관 {s['hold1_steps']:.1f}행동·{s['hold1_days']:.2f}일, 최대 보유 {s['max_held1']:.2f}, "
            f"같은 등급 2개+ 보유 {s['hold2_frac']:.1%}(게임 {s['games_hold2']:.0%}) 3개+ {s['hold3_frac']:.2%}(게임 {s['games_hold3']:.0%}), "
            f"합성 기회 {s['merge_opps']:.2f}·선택 {s['merge_take']:.0%}, 합성 우선 규칙 {s['merge_rule']:.2f} | "
            f"합성 은/금상자 개봉 {s['reinvest3']:.3f}/{s['reinvest4']:.3f} 재투자 보상 {s['r_reinvest']:.3f}")


def parse_emergency(text):
    """'등급,스왑 상한,하트 상한' → (int, int, int), 비우면 None"""
    return tuple(int(x) for x in text.split(",")) if text else None


def masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    N, T = args.num_envs, args.num_steps
    chest_bonus = [float(x) for x in args.chest_bonus.split(",")] if args.chest_bonus else None
    env = ts.VecEnv(N, seed=args.seed, threads=args.threads, chest_bonus=chest_bonus,
                    econ_w=args.econ_shaping, econ_gamma=args.gamma,
                    no_toss_day_off=bool(args.no_toss_day_off), no_open_normal=bool(args.no_open_normal),
                    open_min_tier=args.open_min_tier, emergency=parse_emergency(args.emergency),
                    merge_rule=tuple(int(x) for x in args.merge_rule.split(",")) if args.merge_rule else None,
                    reinvest_bonus=tuple(float(x) for x in args.reinvest_bonus.split(",")) if args.reinvest_bonus else None,
                    no_toss=bool(args.no_toss), no_cannon_flip=bool(args.no_cannon_flip))
    levels = [int(x) for x in args.start_level.split(",")] if args.start_level else []
    aux_levels = [int(x) for x in args.aux_levels.split(",")] if args.aux_coef > 0 else []
    if levels:
        t0 = time.time()
        for i, lv in enumerate(levels):
            env.build_start_pool(args.start_pool // len(levels), lv, seed=args.seed + 1000 * lv,
                                 confirm_samples=args.aux_confirm if lv in aux_levels else 0, append=i > 0)
        env.set_start_frac(args.start_frac)
        print(f"연습 시작 상태 {env.start_pool_info()} (난이도 {levels}, {time.time() - t0:.0f}s), 비율 {args.start_frac}")
    if args.silver_starts:
        from search_eval import Policy
        from starts import capture, describe
        t0 = time.time()
        pool_pol = Policy(args.pool_ckpt or args.init, dev)
        hold, k, info = capture(pool_pol, args.pool_games, args.pool_seed, args.silver_starts,
                                env_kw={"open_min_tier": args.pool_open_min_tier})
        added = env.add_starts(hold, np.arange(k, dtype=np.int64), SILVER_LEVEL, place_tier=3, seed=args.pool_seed)
        env.set_start_frac(args.start_frac)
        print(f"은상자 연습 시작 상태 {added}개: {describe(info)} ({time.time() - t0:.0f}s), 비율 {args.start_frac}", flush=True)
        if args.stage_starts:
            added2 = env.add_starts_layout(hold, np.arange(k, dtype=np.int64), STAGE_LEVEL[args.stage_starts], args.stage_starts,
                                           seed=args.pool_seed)
            print(f"동상자 연습({args.stage_starts}, 난이도 {STAGE_LEVEL[args.stage_starts]}) 시작 상태 {added2}개 "
                  f"(연습 비율 {args.start_frac}를 은상자 연습과 나눔)", flush=True)
        del pool_pol, hold
    grid = np.zeros((N, C, H, W), np.float32)
    scal = np.zeros((N, S), np.float32)
    mask = np.zeros((N, A), bool)
    rew = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    days = np.zeros(N, np.float32)
    env.reset(grid, scal, mask)

    init = torch.load(args.init, map_location=dev) if args.init else None
    if init:
        args.channels, args.blocks, args.hidden = (init["args"][k] for k in ("channels", "blocks", "hidden"))
    agent = Agent(args.channels, args.blocks, args.hidden).to(dev)
    if init:
        agent.load_state_dict(init["agent"])
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
    b_aux = torch.zeros((T, N), dtype=torch.bool, device=dev)  # 보조 손실 대상 여부
    b_auxt = torch.zeros((T, N, A), dtype=torch.bool, device=dev)  # 합성 행동 집합 M(s)
    aux_flag, aux_tgt = np.zeros(N, bool), np.zeros((N, A), bool)
    aux_decay_from = None  # 계수를 줄이기 시작한 업데이트

    n_updates = args.total_steps // (N * T)
    log_f = open(os.path.join(run_dir, "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["update", "step", "sps"] + SUMMARY_KEYS + PRACTICE_KEYS +
                     ["pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "explained_var", "aux_coef", "aux_merge_p", "aux_frac"])
    t_start, step0 = time.time(), global_step
    ep_hist = []
    rstd = RunningStd(N)
    for ck in (init, torch.load(args.resume, map_location="cpu") if args.resume else None):
        if ck and "rstd" in ck:
            rstd.mean, rstd.var, rstd.count = ck["rstd"]
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
            if args.aux_coef > 0:
                env.aux_targets(aux_flag, aux_tgt)
                b_aux[t] = torch.from_numpy(aux_flag).to(dev)
                b_auxt[t] = torch.from_numpy(aux_tgt).to(dev)
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
        # 보조 손실 계수: 연습 게임의 합성 선택률이 목표를 넘으면 aux_decay 업데이트에 걸쳐 0으로
        aux_c = args.aux_coef
        if aux_c > 0 and aux_decay_from is not None:
            aux_c *= max(0.0, 1.0 - (update - aux_decay_from) / args.aux_decay)
        fx, fxt = b_aux.reshape(-1), b_auxt.reshape(-1, A)
        aux_frac = fx.float().mean().item()  # K/B: 보조 손실 대상 샘플 비율
        aux_stats = []
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
                if update <= args.value_warmup:
                    loss = args.vf_coef * v_loss
                else:
                    loss = pg + args.vf_coef * v_loss - args.ent_coef * ent
                    sel = fx[idx]
                    if aux_c > 0 and bool(sel.any()):
                        lp = logits[sel].masked_fill(~fm[idx][sel], -1e9).log_softmax(1)
                        merge_lp = lp.masked_fill(~fxt[idx][sel], -1e9).logsumexp(1)  # log Σ_{a∈M} π(a|s)
                        # 미니배치 전체 크기로 나눈다: 대상 샘플 하나의 무게 = 어드밴티지 1인 정책 경사 샘플 × 계수
                        l_aux = -merge_lp.sum() / len(idx)
                        loss = loss + aux_c * l_aux
                        aux_stats.append(merge_lp.detach().exp().mean())
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
        s = episode_summary([f for f in fin if not f["practice"]])
        aux_p = torch.stack(aux_stats).mean().item() if aux_stats else float("nan")
        if args.aux_coef > 0 and aux_decay_from is None:
            top = max(aux_levels)
            pr = practice_summary(ep_hist, top)
            if pr and pr["episodes"] >= 200 and pr["success_path"] >= args.aux_target:
                aux_decay_from = update
                print(f"연습(난이도 {top}) 성공률 {pr['success_path']:.0%} ≥ {args.aux_target:.0%}: 보조 손실 계수를 줄이기 시작", flush=True)
        log.writerow([update, global_step, int(sps)] + [s[k] if s else "" for k in SUMMARY_KEYS] +
                     practice_row(fin) + st + [ev, aux_c, aux_p, aux_frac])
        log_f.flush()
        if update % 5 == 0 or update == 1:
            pr = ""
            for lv in (1, 2, 3, 4):
                ps = practice_summary(ep_hist, lv)
                if ps:
                    pr += (f" | 연습{lv} {ps['episodes']}게임 성공 {ps['success_path']:.0%}(당일 {ps['success_day']:.0%}) "
                           f"기회만 {ps['ready_no_merge']:.0%} 동 {ps['made2']:.0%}")
            sv = [f for f in ep_hist if f["practice_level"] == SILVER_LEVEL]
            if sv:
                op = np.array([f["opened"][2] + f["opened"][3] for f in sv])
                pr += (f" | 은상자 연습 {len(sv)}게임 개봉 {np.mean(op >= 1):.0%}·둘째 {np.mean(op >= 2):.0%}·셋째 {np.mean(op >= 3):.0%} "
                       f"이후 생존 {np.mean([f['day'] - f['start_day'] for f in sv]):.1f}일 "
                       f"합성 우선 {np.mean([f['merge_rule_uses'] for f in sv]):.1f} 재투자 보상 {np.mean([f['r_reinvest'] for f in sv]):.2f}")
            for lay, lv in STAGE_LEVEL.items():
                bz = [f for f in ep_hist if f["practice_level"] == lv]
                if bz:
                    pr += (f" | 동상자 연습({lay}) {len(bz)}게임 은상자 합성 {np.mean([f['made'][2] > 0 for f in bz]):.0%}·"
                           f"합성 은상자 개봉 {np.mean([f['reinvest_opens'][2] > 0 for f in bz]):.0%} 비상 개봉 "
                           f"{np.mean([f['emergency_opens'] for f in bz]):.2f} 이후 생존 {np.mean([f['day'] - f['start_day'] for f in bz]):.1f}일 "
                           f"재투자 보상 {np.mean([f['r_reinvest'] for f in bz]):.2f}")
            print(f"[{update}/{n_updates}] step {global_step:,} sps {sps:,.0f} (롤아웃 {t_roll:.1f}s) | 정상 시작 "
                  f"{format_summary(episode_summary([f for f in ep_hist if not f['practice']]))}{pr} | "
                  f"pg {st[0]:.4f} v {st[1]:.4f} ent {st[2]:.3f} kl {st[3]:.4f} ev {ev:.3f} rstd {rstd.std:.4f}"
                  + (f" | 보조 계수 {aux_c:.3f} 대상 비율 {aux_frac:.4f} 실효 {aux_c * aux_frac:.4f} 대상 확률 {aux_p:.3f}"
                     if args.aux_coef > 0 else ""), flush=True)
        keep = args.keep_every and update % args.keep_every == 0
        if update % args.save_every == 0 or update == n_updates or keep:
            ck = {"agent": agent.state_dict(), "opt": opt.state_dict(), "update": update,
                  "global_step": global_step, "args": vars(args), "rstd": (rstd.mean, rstd.var, rstd.count)}
            torch.save(ck, os.path.join(run_dir, "agent.pt"))
            if keep:
                torch.save(ck, os.path.join(run_dir, f"agent_{round(global_step / 1e6)}M.pt"))


if __name__ == "__main__":
    main()
