"""계획 선택기 모델(공용): 원본 정책 인코더(Agent.trunk, 동결) + 후보별 작은 MLP → (교사 기준 통과 로짓, 기준 행동 대비 ΔQ).
입력은 실행 시 얻을 수 있는 것만: 현재 관측, 원본 정책이 고른 행동(기준 행동)과 그 가상 결과, 후보의 첫·둘째 수,
후보의 가상 결과(공격 무기 생성 수, 결과 무기 종류·등급·위치, 사라진 무기, 자원 칸 변화, 사용 스왑), 정책 로그 확률.
밤 피해·다음 아침 하트·교사 Q·통과 판정은 입력에 없다(정답으로만 쓴다)."""
import numpy as np
import torch
import torch.nn as nn

from ppo import Agent

NC, NCR = 6, 6
DX = np.array([0, 0, -1, 1])
DY = np.array([-1, 1, 0, 0])


def drag_cells(a):
    """드래그 행동 → (출발 행, 출발 열0, 도착 행, 도착 열0, 유효). 격자 행 0 = 성 줄, 1..7 = 보드"""
    a = np.asarray(a)
    valid = (a >= 0) & (a < 168)
    aa = np.where(valid, a, 0)
    c, d = aa // 4, aa % 4
    x0, y = c % 6, c // 6 + 1
    return y, x0, np.clip(y + DY[d], 0, 7), np.clip(x0 + DX[d], 0, 5), valid


def cr_summary(cr):
    """결과 무기 [..., NCR, 4](종류, 등급, x, y; 없으면 -1) → 종류×등급 수 12, 행별 수 7, 평균 열"""
    cr = cr.astype(np.int64)
    ok = cr[..., 0] >= 0
    att = ok & (cr[..., 0] < 3)
    kt = np.zeros(cr.shape[:-2] + (12,), np.float32)
    for k in range(3):
        for t in range(1, 5):
            kt[..., k * 4 + t - 1] = ((cr[..., 0] == k) & (cr[..., 1] == t) & att).sum(-1)
    rows = np.stack([((cr[..., 3] == r) & ok).sum(-1) for r in range(1, 8)], -1).astype(np.float32)
    n = np.maximum(ok.sum(-1), 1)
    mx = (np.where(ok, cr[..., 2], 0).sum(-1) / n / 6.0).astype(np.float32)
    return np.concatenate([kt, rows, mx[..., None]], -1)


def features(d):
    """결정 묶음(dict of arrays, planner_data 형식) → 손 특징 [B, NC, F], 칸 위치 [B, NC, 7, 2](행, 열0; 없으면 -1), 후보 유효 [B, NC]"""
    B = len(d["base"])
    k = np.asarray(d["k"])
    cvalid = np.arange(NC)[None, :] < k[:, None]
    b_sig = d["b_sig"].astype(np.float32)
    b_cr = cr_summary(d["b_cr"])
    b_lost = d["b_lost"].astype(np.float32)
    base = np.asarray(d["base"])
    bcat = np.stack([base < 168, (base >= 168) & (base < 216), base >= 216], -1).astype(np.float32)
    bf = np.concatenate([b_sig, b_cr, b_lost, np.stack([d["b_res"], d["b_dsw"], d["b_ok"], d["b_same"],
                                                        np.clip(d["lp_base"], -30, 0) / 10], -1).astype(np.float32), bcat], -1)
    st = np.stack([np.asarray(d["hearts"]) / 30.0, np.asarray(d["day"]) / 30.0, np.asarray(d["swaps"]) / 70.0], -1).astype(np.float32)
    c_sig = d["c_sig"].astype(np.float32)
    c_cr = cr_summary(d["c_cr"])
    c_lost = d["c_lost"].astype(np.float32)
    two = (np.asarray(d["a2"]) >= 0).astype(np.float32)
    lp1 = np.clip(np.nan_to_num(np.asarray(d["lp1"], np.float32), nan=-30.0), -30, 0) / 10
    dir1 = np.eye(4, dtype=np.float32)[np.where(np.asarray(d["a1"]) >= 0, np.asarray(d["a1"]) % 4, 0)]
    dir2 = np.eye(4, dtype=np.float32)[np.where(np.asarray(d["a2"]) >= 0, np.asarray(d["a2"]) % 4, 0)] * two[..., None]
    cf = np.concatenate([c_sig, c_cr, c_lost, np.asarray(d["c_res"], np.float32)[..., None], two[..., None], lp1[..., None], dir1, dir2,
                         c_sig - b_sig[:, None], (np.asarray(d["c_res"], np.float32) - np.asarray(d["b_res"], np.float32)[:, None])[..., None]], -1)
    hand = np.concatenate([cf, np.repeat(bf[:, None], NC, 1), np.repeat(st[:, None], NC, 1)], -1) * cvalid[..., None]
    cells = np.full((B, NC, 7, 2), -1, np.int64)
    for j, a in enumerate((d["a1"], d["a2"])):
        y, x0, ty, tx, v = drag_cells(a)
        cells[..., 2 * j, 0], cells[..., 2 * j, 1] = np.where(v, y, -1), np.where(v, x0, -1)
        cells[..., 2 * j + 1, 0], cells[..., 2 * j + 1, 1] = np.where(v, ty, -1), np.where(v, tx, -1)
    cr0 = np.asarray(d["c_cr"])[:, :, 0].astype(np.int64)
    cells[..., 4, 0], cells[..., 4, 1] = np.where(cr0[..., 0] >= 0, cr0[..., 3], -1), np.where(cr0[..., 0] >= 0, cr0[..., 2] - 1, -1)
    y, x0, ty, tx, v = drag_cells(base)
    for j, (yy, xx) in enumerate(((y, x0), (ty, tx))):
        cells[:, :, 5 + j, 0] = np.where(v, yy, -1)[:, None]
        cells[:, :, 5 + j, 1] = np.where(v, xx, -1)[:, None]
    cells[~cvalid] = -1
    return hand.astype(np.float32), cells, cvalid


class Selector(nn.Module):
    def __init__(self, enc_args, d_hand, hidden=256):
        super().__init__()
        self.enc = Agent(enc_args["channels"], enc_args["blocks"], enc_args["hidden"])
        for q in self.enc.parameters():
            q.requires_grad_(False)
        ch, hd = enc_args["channels"], enc_args["hidden"]
        self.register_buffer("mu", torch.zeros(d_hand))
        self.register_buffer("sd", torch.ones(d_hand))
        self.mlp = nn.Sequential(nn.Linear(hd + 7 * ch + d_hand, hidden), nn.ReLU(), nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 2))

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()  # 인코더는 항상 평가 모드·동결
        return self

    def forward(self, grid, scal, hand, cells):
        with torch.no_grad():
            x, h = self.enc.trunk(grid, scal)  # x [B, ch, 8, 6], h [B, hidden]
        B, ch = x.shape[:2]
        flat = x.flatten(2).transpose(1, 2)  # [B, 48, ch]
        ok = cells[..., 0] >= 0  # [B, NC, 7]
        idx = (cells[..., 0].clamp(min=0) * 6 + cells[..., 1].clamp(min=0)).reshape(B, -1)  # [B, NC*7]
        g = torch.gather(flat, 1, idx[..., None].expand(-1, -1, ch)).reshape(B, NC, 7, ch) * ok[..., None]
        z = torch.cat([h[:, None].expand(-1, NC, -1), g.flatten(2), (hand - self.mu) / self.sd], -1)
        out = self.mlp(z)
        return out[..., 0], out[..., 1]  # 통과 로짓, ΔQ(표준화 단위)


def load_selector(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Selector(ck["enc_args"], ck["d_hand"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck
