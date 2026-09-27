//! 경제 조형 검증: Φ(s) = w × (남은 스왑 + 보드 상자의 개봉 스왑), 보상에 γ^days·Φ(s′) − Φ(s).
//! 합성 한 번의 조형 보상과, 한 게임 전체에서 할인 합이 −Φ(s0)로 소거되는지 확인한다.
use towerswap_core::env::{Env, N_ACTIONS};
use towerswap_core::expert::greedy_action;
use towerswap_core::kinds::Kind;
use towerswap_core::{Game, Phase};

/// tests/chest_merge.rs와 같은 보드: (4,1)의 상자를 왼쪽으로 끌면 일반 상자 세 개가 합쳐진다
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

fn chest_units(g: &Game) -> i64 {
    const OPEN: [i64; 5] = [0, 2, 12, 70, 380];
    (1..=6)
        .flat_map(|x| (1..=7).map(move |y| (x, y)))
        .filter_map(|(x, y)| g.grid[x][y])
        .filter(|&t| g.tiles[t].kind == Kind::Chest)
        .map(|t| OPEN[g.tiles[t].tier.min(4) as usize])
        .sum()
}

#[test]
fn merge_gives_economic_gain() {
    let mut game = Game::new_game(7, 11);
    game.load_board(BOARD);
    let mut env = Env::from_game(game);
    env.econ_w = 0.01;
    env.econ_gamma = 0.995;
    let (sw0, ch0) = (env.game.swaps, chest_units(&env.game));
    assert_eq!(ch0, 6);
    let out = env.step(14).expect("합성 드래그가 무효");
    let (sw1, ch1) = (env.game.swaps, chest_units(&env.game));
    assert_eq!(ch1, 12, "동상자 하나만 남아야 한다");
    assert_eq!(out.days, 0);
    // 점수 변화 없음 → 보상 = 조형 = w × (스왑 변화 + 상자 +6)
    let want = 0.01 * ((sw1 - sw0) + (ch1 - ch0)) as f32;
    assert!((out.reward - want).abs() < 1e-6, "보상 {} 기대 {want} (스왑 {sw0} → {sw1})", out.reward);
    assert!(out.reward > 0.0);
}

/// 한 게임의 할인 조형 합: Σ γ^{D_t}·(γ^{d_t}Φ_{t+1} − Φ_t) = γ^{D_T}Φ_T − Φ_0 = −Φ_0 (게임 오버 Φ = 0)
#[test]
fn shaping_telescopes_over_a_game() {
    for (seed, gamma) in [(3u32, 1.0f32), (5, 0.9)] {
        let mut env = Env::new(seed, seed + 100);
        env.econ_w = 1.0;
        env.econ_gamma = gamma;
        let phi0 = (env.game.swaps.max(0) + chest_units(&env.game)) as f64;
        let (mut total, mut disc, mut rng) = (0f64, 1f64, seed as u64);
        let mut mask = vec![false; N_ACTIONS];
        for _ in 0..200_000 {
            if env.game.phase == Phase::GameOver {
                break;
            }
            env.mask(&mut mask);
            let a = greedy_action(&env, &mask, &mut rng);
            let s0 = env.game.score();
            let out = env.step(a).unwrap();
            let f = out.reward as f64 - (env.game.score() - s0) as f64 / 100.0;
            total += disc * f;
            disc *= (gamma as f64).powi(out.days as i32);
        }
        assert_eq!(env.game.phase, Phase::GameOver);
        assert!((total + phi0).abs() < 1e-2 * phi0.max(1.0), "γ {gamma}: 합 {total}, −Φ0 {}", -phi0);
        if gamma == 1.0 {
            assert!((env.ep_econ as f64 + phi0).abs() < 1e-1, "ep_econ {}", env.ep_econ);
        }
    }
}
