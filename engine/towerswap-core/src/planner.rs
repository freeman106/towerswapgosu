//! 계획 휴리스틱: 한 수 greedy에 짧은 계획을 더한다. 상자가 걸린 결정(보드에 1~3등급 상자, 또는 밤 직전에 상자)에서만 쓴다.
//!
//! 1. 빔 탐색(최대 3수)으로 행동열을 만든다. 정적 평가에 합성 거리 프리미엄이 있어 점수가 당장 떨어지는 준비 이동도 남는다.
//!    빔에는 "합성을 포함한 최선", "개봉을 포함한 최선", "둘 다 없는 최선"을 칸별로 하나씩 꼭 남긴다.
//! 2. 첫 행동이 다른 후보 몇 개를 고르고, 각 행동열 뒤를 두 뒤이음 정책(개봉형 = 기본 greedy, 보관형 = `Params::hold`)으로
//!    그날 밤이 끝날 때까지 굴린다. 모든 후보가 같은 가상 난수 묶음을 쓴다(실제 게임의 미래는 보지 않는다).
//! 3. 끝 상태를 생존과 투자 가치로 평가한다: 사망 벌점, 타워 + 하트, λ·(스왑 + 상자 개봉 스왑 + 합성 프리미엄).
//!    프리미엄은 같은 등급이 3개 이상이면 합성 거리에 따라, 2개면 작게만 준다(외딴 상자는 개봉 가치만).
//! 4. 가장 좋은 후보의 첫 행동만 실행하고 다음 결정에서 다시 계획한다.
//!
//! 파라미터는 환경변수 TS_PLAN="beam=6,depth=3,..."로 바꿀 수 있다.
use crate::analysis::Board;
use crate::env::{Env, A_END_DAY, N_ACTIONS};
use crate::expert::{greedy_action, greedy_set_with, heuristic_with, lcg, params, skipped, Params};
use crate::game::{Game, Phase};
use crate::items::chest_swaps;
use crate::rng::JsRng;

#[derive(Clone)]
pub struct PlanParams {
    pub beam: usize,    // 깊이마다 남기는 행동열 수 (칸별 최선은 별도)
    pub depth: usize,   // 행동열 최대 길이
    pub cands: usize,   // 롤아웃으로 비교할 첫 행동 후보 수 (칸별 최선은 별도)
    pub seeds: u32,     // 가상 난수 묶음 수
    pub lambda: f64,    // 스왑 1개의 가치 (타워 가치 단위, 기본 greedy에서 잰 평균 약 0.24)
    pub near: [f64; 3], // 합성 프리미엄 실현 비율: 같은 등급 3개 이상에서 합성 거리 1, 2, 그보다 멂
    pub pair: f64,      // 같은 등급 2개의 프리미엄 비율
    pub death: f64,     // 사망 벌점 (타워 가치 단위)
}

impl Default for PlanParams {
    fn default() -> Self {
        PlanParams { beam: 6, depth: 3, cands: 4, seeds: 6, lambda: 0.2, near: [0.8, 0.6, 0.4], pair: 0.2, death: 1000.0 }
    }
}

pub fn plan_params() -> &'static PlanParams {
    static P: std::sync::OnceLock<PlanParams> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut p = PlanParams::default();
        for kv in std::env::var("TS_PLAN").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("TS_PLAN 형식: k=v,...");
            let v: f64 = v.parse().expect("TS_PLAN 값");
            match k {
                "beam" => p.beam = v as usize,
                "depth" => p.depth = v as usize,
                "cands" => p.cands = v as usize,
                "seeds" => p.seeds = v as u32,
                "lambda" => p.lambda = v,
                "near1" => p.near[0] = v,
                "near2" => p.near[1] = v,
                "far" => p.near[2] = v,
                "pair" => p.pair = v,
                "death" => p.death = v,
                _ => panic!("TS_PLAN: 모르는 키 {k}"),
            }
        }
        p
    })
}

/// 등급 k 상자 3개 → k+1 1개로 늘어나는 개봉 스왑 (6, 34, 170)
const GAIN: [f64; 4] = [0.0, 6.0, 34.0, 170.0];

/// 계획 평가(타워 가치 단위): 사망 벌점, 타워 + 하트, λ·(스왑 + 상자 개봉 스왑 + 합성 프리미엄)
pub fn plan_value(g: &Game, pp: &PlanParams) -> f64 {
    if g.phase == Phase::GameOver {
        return -pp.death;
    }
    let b = Board::of(g);
    let n = b.chest_counts();
    let open: i64 = (1..=4).map(|k| n[k] as i64 * chest_swaps(k as u8)).sum();
    let (d1, d2) = if (1..=3).any(|k| n[k] >= 3) { b.chest_merge_bits() } else { (0, 0) };
    let mut premium = 0.0;
    for k in 1..=3 {
        let f = if n[k] >= 3 {
            if d1 & (1 << k) != 0 {
                pp.near[0]
            } else if d2 & (1 << k) != 0 {
                pp.near[1]
            } else {
                pp.near[2]
            }
        } else if n[k] == 2 {
            pp.pair
        } else {
            0.0
        };
        premium += f * GAIN[k];
    }
    heuristic_with(g, &Params::towers_only()) + pp.lambda * ((g.swaps.max(0) + open) as f64 + premium)
}

/// 상자가 걸린 결정인지: 입력 대기(낮·밤 직전), 휴식일이 아님, 보드에 1~3등급 상자(밤 직전이면 아무 상자)
pub fn chest_decision(g: &Game) -> bool {
    if !matches!(g.phase, Phase::Idle | Phase::Dusk) || g.day == g.day_off_day {
        return false;
    }
    let n = Board::of(g).chest_counts();
    n[1..4].iter().any(|&c| c > 0) || (g.phase == Phase::Dusk && n[4] > 0)
}

struct Node {
    env: Env,
    path: Vec<usize>,
    score: f64,
    merged: bool, // 행동열에 상자 합성이 있다
    opened: bool, // 행동열에 상자 개봉이 있다
}

fn merged_count(g: &Game) -> u32 {
    g.chest_made[2..].iter().sum()
}

fn opened_count(g: &Game) -> u32 {
    g.chest_opened[1..].iter().sum()
}

/// 칸 번호: 0 합성 포함, 1 개봉 포함(합성 없음), 2 둘 다 없음
fn bucket(n: &Node) -> usize {
    if n.merged {
        0
    } else if n.opened {
        1
    } else {
        2
    }
}

/// 빔 탐색에서 나온 행동열 (환경 없이)
struct Plan {
    path: Vec<usize>,
    score: f64,
    bucket: usize,
}

/// 점수순 상위 `beam`개 + 칸별 최선을 남긴다 (`nodes`는 점수 내림차순으로 정렬돼 있어야 한다)
fn select(nodes: Vec<Node>, beam: usize) -> Vec<Node> {
    let mut keep = vec![false; nodes.len()];
    let mut seen = [false; 3];
    for (i, n) in nodes.iter().enumerate() {
        let b = bucket(n);
        if i < beam || !seen[b] {
            keep[i] = true;
        }
        seen[b] = true;
    }
    nodes.into_iter().zip(keep).filter(|(_, k)| *k).map(|(n, _)| n).collect()
}

/// 빔 탐색: 모든 깊이에서 남은 행동열을 돌려준다
fn beam_search(env: &Env, pp: &PlanParams, seeds: (u32, u32)) -> Vec<Plan> {
    let mut root = env.clone();
    root.game.rng_v = JsRng::new(seeds.0);
    root.game.rng_m = JsRng::new(seeds.1);
    let (m0, o0) = (merged_count(&root.game), opened_count(&root.game));
    let mut frontier = vec![Node { env: root, path: Vec::new(), score: 0.0, merged: false, opened: false }];
    let mut all = Vec::new();
    let mut mask = [false; N_ACTIONS];
    for _ in 0..pp.depth {
        let mut children = Vec::new();
        for node in &frontier {
            let g = &node.env.game;
            if !matches!(g.phase, Phase::Idle | Phase::Dusk) {
                continue; // 선택 창 등은 더 펼치지 않는다
            }
            node.env.mask(&mut mask);
            for a in 0..N_ACTIONS {
                if !mask[a] || a == A_END_DAY || skipped(g, a) {
                    continue;
                }
                let mut e = node.env.clone();
                e.step(a).unwrap();
                let mut path = node.path.clone();
                path.push(a);
                let score = plan_value(&e.game, pp);
                let (merged, opened) = (merged_count(&e.game) > m0, opened_count(&e.game) > o0);
                children.push(Node { env: e, path, score, merged, opened });
            }
        }
        if children.is_empty() {
            break;
        }
        children.sort_by(|a, b| b.score.total_cmp(&a.score));
        frontier = select(children, pp.beam);
        all.extend(frontier.iter().map(|n| Plan { path: n.path.clone(), score: n.score, bucket: bucket(n) }));
    }
    all
}

/// 행동열 `path` 뒤를 정책 `p`로 그날 밤이 끝날 때까지 굴린 끝 상태의 평가.
/// 다른 난수에서 무효가 된 행동을 만나면 거기서 행동열을 끊고 정책으로 이어 간다.
fn rollout_value(env: &Env, path: &[usize], p: &Params, pp: &PlanParams, seeds: (u32, u32), tie: u64) -> f64 {
    let mut e = env.clone();
    e.game.rng_v = JsRng::new(seeds.0);
    e.game.rng_m = JsRng::new(seeds.1);
    let day = e.game.day;
    for &a in path {
        if e.done() || e.game.day != day || e.step(a).is_none() {
            break;
        }
    }
    let (mut m, mut set, mut rng) = ([false; N_ACTIONS], [false; N_ACTIONS], tie);
    for _ in 0..20_000 {
        if e.done() || e.game.day != day {
            break;
        }
        e.mask(&mut m);
        let a = greedy_set_with(&e, &m, &mut set, &mut rng, p);
        e.step(a).unwrap();
    }
    plan_value(&e.game, pp)
}

/// 계획 휴리스틱의 행동. 상자가 걸리지 않은 결정은 한 수 greedy.
pub fn plan_action(env: &Env, mask: &[bool], rng: &mut u64) -> usize {
    if !chest_decision(&env.game) {
        return greedy_action(env, mask, rng);
    }
    let pp = plan_params();
    let seeds = |r: &mut u64| (lcg(r) as u32 | 1, lcg(r) as u32 | 1);
    let nodes = beam_search(env, pp, seeds(rng));

    // 첫 행동별 최선의 행동열
    let mut best: Vec<(usize, usize)> = Vec::new(); // (첫 행동, nodes 번호)
    for (i, n) in nodes.iter().enumerate() {
        match best.iter_mut().find(|(a, _)| *a == n.path[0]) {
            Some(b) if nodes[b.1].score < n.score => b.1 = i,
            Some(_) => {}
            None => best.push((n.path[0], i)),
        }
    }
    best.sort_by(|a, b| nodes[b.1].score.total_cmp(&nodes[a.1].score));
    let mut cands: Vec<Vec<usize>> = Vec::new();
    let mut seen = [false; 3];
    for (k, &(_, i)) in best.iter().enumerate() {
        let b = nodes[i].bucket;
        if k < pp.cands || !seen[b] {
            cands.push(nodes[i].path.clone());
        }
        seen[b] = true;
    }
    if mask[A_END_DAY] {
        cands.push(vec![A_END_DAY]);
    }
    if cands.len() == 1 {
        return cands[0][0];
    }

    // 모든 후보·정책이 같은 가상 난수 묶음과 동점 깨기 난수를 쓴다
    let draws: Vec<((u32, u32), u64)> = (0..pp.seeds).map(|_| (seeds(rng), lcg(rng))).collect();
    let (open_p, hold_p) = (params(), Params::hold());
    let mut choice = (f64::NEG_INFINITY, cands[0][0]);
    for path in &cands {
        let mean = |p: &Params| draws.iter().map(|&(s, t)| rollout_value(env, path, p, pp, s, t)).sum::<f64>() / pp.seeds as f64;
        let v = mean(open_p).max(mean(&hold_p));
        if v > choice.0 {
            choice = (v, path[0]);
        }
    }
    choice.1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tests/chest_merge.rs의 보드: 1행 h1 h1 s0 h1 — (4,1)을 왼쪽으로 끌면 일반 상자 세 개가 합쳐진다
    const BOARD: &str = concat!(
        "e0e0e0e0e0e0",
        "h1h1s0h1l0i0",
        "l0i0g0d0s0l0",
        "s0g0d0l0i0s0",
        "i0d0l0s0g0d0",
        "g0l0i0g0d0i0",
        "d0s0g0i0l0g0",
        "o0o0o0o0o0o0",
    );

    fn env_of(board: &str) -> Env {
        let mut game = Game::new_game(7, 11);
        game.load_board(board);
        Env::from_game(game)
    }

    #[test]
    fn premium_depends_on_merge_distance() {
        let pp = PlanParams::default();
        let near = env_of(BOARD);
        let far = env_of(&BOARD.replacen("h1h1s0h1l0i0", "h1s0h1l0h1i0", 1));
        let base = |e: &Env| heuristic_with(&e.game, &Params::towers_only()) + pp.lambda * (e.game.swaps + 6) as f64;
        let prem = |e: &Env| (plan_value(&e.game, &pp) - base(e)) / pp.lambda;
        assert!((prem(&near) - pp.near[0] * 6.0).abs() < 1e-9, "거리 1 프리미엄 {}", prem(&near));
        assert!((prem(&far) - pp.near[1] * 6.0).abs() < 1e-9, "거리 2 프리미엄 {}", prem(&far));
    }

    /// 교환 두 번이면 합쳐지는 보드에서 빔에 "준비 이동 → 합성" 행동열이 남는다
    #[test]
    fn beam_keeps_prep_then_merge() {
        let env = env_of(&BOARD.replacen("h1h1s0h1l0i0", "h1s0h1l0h1i0", 1));
        let plans = beam_search(&env, &PlanParams::default(), (3, 5));
        let merge = plans.iter().filter(|p| p.bucket == 0).map(|p| p.path.len()).min();
        assert_eq!(merge, Some(2), "두 수 합성 경로가 빔에 없다");
    }
}
