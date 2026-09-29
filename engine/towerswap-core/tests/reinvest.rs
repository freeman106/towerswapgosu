//! 재투자 보상 검증: 합성으로 만든 은상자를 열 때만 보상이 붙고(보드 불러오기·연습 배치로 놓인 은상자는 제외),
//! 보상은 행동 보상과 ep_reinvest에 더해지며 재투자 개봉 횟수가 오른다.
use towerswap_core::env::{Env, A_CELL};
use towerswap_core::kinds::Kind;
use towerswap_core::Game;

/// 1행: h2 h2 s0 h2 l0 i0 — (4,1)의 동상자를 왼쪽으로 끌면 동상자 세 개가 모여 은상자가 된다.
const MERGE_BOARD: &str = concat!(
    "e0e0e0e0e0e0",
    "h2h2s0h2l0i0",
    "l0i0g0d0s0l0",
    "s0g0d0l0i0s0",
    "i0d0l0s0g0d0",
    "g0l0i0g0d0i0",
    "d0s0g0i0l0g0",
    "o0o0o0o0o0o0",
);

/// 1행 1열에 처음부터 은상자가 놓인 보드
const PLACED_BOARD: &str = concat!(
    "e0e0e0e0e0e0",
    "h3l0s0i0l0i0",
    "l0i0g0d0s0l0",
    "s0g0d0l0i0s0",
    "i0d0l0s0g0d0",
    "g0l0i0g0d0i0",
    "d0s0g0i0l0g0",
    "o0o0o0o0o0o0",
);

const BONUS: [f32; 5] = [0.0, 0.0, 0.0, 3.0, 16.3];

fn find_chest(g: &Game, tier: u8) -> Option<(i32, i32)> {
    (1..=7).flat_map(|y| (1..=6).map(move |x| (x, y))).find(|&(x, y)| {
        g.grid[x as usize][y as usize].map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier == tier)
    })
}

fn tap_action(x: i32, y: i32) -> usize {
    A_CELL + (y as usize) * 6 + (x as usize - 1)
}

#[test]
fn merged_silver_pays_bonus() {
    let mut game = Game::new_game(7, 11);
    game.load_board(MERGE_BOARD);
    let mut env = Env::from_game(game);
    env.reinvest_bonus = BONUS;
    let merge = env.step(14).expect("합성 드래그가 무효");
    assert!(merge.reward < 1.0, "합성 자체에는 재투자 보상이 없다");
    let (x, y) = find_chest(&env.game, 3).expect("은상자가 생겨야 한다");
    let t = env.game.grid[x as usize][y as usize].unwrap();
    assert!(env.game.tiles[t].chest_merged, "합성으로 만든 은상자 표시가 없다");
    let out = env.step(tap_action(x, y)).expect("은상자 탭이 무효");
    assert!((out.reward - 3.0).abs() < 1e-4, "재투자 보상 3이어야 한다: {}", out.reward);
    assert!((env.ep_reinvest - 3.0).abs() < 1e-4);
    assert_eq!(env.reinvest_opens[3], 1);
}

#[test]
fn placed_silver_pays_nothing() {
    // 보드 불러오기로 놓인 은상자
    let mut game = Game::new_game(7, 11);
    game.load_board(PLACED_BOARD);
    let mut env = Env::from_game(game);
    env.reinvest_bonus = BONUS;
    let (x, y) = find_chest(&env.game, 3).expect("은상자가 있어야 한다");
    let out = env.step(tap_action(x, y)).expect("은상자 탭이 무효");
    assert!(out.reward.abs() < 1e-4, "놓인 은상자에는 재투자 보상이 없다: {}", out.reward);
    assert_eq!(env.reinvest_opens[3], 0);
    assert_eq!(env.ep_reinvest, 0.0);

    // 연습 시작처럼 자원 하나를 은상자로 바꾼 경우
    let mut game = Game::new_game(7, 11);
    game.load_board(MERGE_BOARD);
    let (x, y) = game.replace_resource_with_chest(3, 0).expect("자원이 있어야 한다");
    let mut env = Env::from_game(game);
    env.reinvest_bonus = BONUS;
    let out = env.step(tap_action(x, y)).expect("은상자 탭이 무효");
    assert!(out.reward.abs() < 1e-4, "연습 배치 은상자에는 재투자 보상이 없다: {}", out.reward);
    assert_eq!(env.reinvest_opens[3], 0);
}

#[test]
fn silver_opens_after_last_swap() {
    // 남은 스왑 1개로 동상자 세 개를 합치면 밤 직전(스왑 0)이 되지만, 은상자부터 여는 규칙에서도 새 은상자는 열 수 있어야 한다
    use towerswap_core::env::{OpenRule, N_ACTIONS};
    use towerswap_core::Phase;
    let mut game = Game::new_game(7, 11);
    game.load_board(MERGE_BOARD);
    game.swaps = 1;
    let mut env = Env::from_game(game);
    env.open_rule = OpenRule { min_tier: 3, emergency: Some((2, 0, 5)) };
    env.step(14).expect("합성 드래그가 무효");
    assert_eq!(env.game.phase, Phase::Dusk, "마지막 스왑을 쓰면 밤 직전이어야 한다");
    let (x, y) = find_chest(&env.game, 3).expect("은상자가 생겨야 한다");
    let mut mask = vec![false; N_ACTIONS];
    env.mask(&mut mask);
    assert!(mask[tap_action(x, y)], "밤 직전에도 새 은상자 개봉이 마스크에서 켜져야 한다");
    env.step(tap_action(x, y)).expect("은상자 개봉이 무효");
    assert!(env.game.swaps > 0, "은상자를 열면 스왑이 생겨야 한다");
}
