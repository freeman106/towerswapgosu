//! 상자 합성 경로 검증: 일반 상자 세 개를 정해진 드래그로 합쳐 동상자를 만들고,
//! 행동 마스크 · 엔진 결과 · 관측(종류·등급 채널) · 생성 카운터 · 최초 생성 보너스가 모두 맞는지 확인한다.
use towerswap_core::analysis::Board;
use towerswap_core::env::{grid_index, kind_index, Env, A_CELL, CH_KIND, CH_TIER, GRID_LEN, N_ACTIONS, N_SCALAR};
use towerswap_core::kinds::Kind;
use towerswap_core::{Game, Phase};

/// 1행: h1 h1 s0 h1 l0 i0 — (4,1)의 상자를 왼쪽으로 끌면 1~3열에 일반 상자 세 개가 모인다.
/// 나머지 칸에는 가로·세로로 같은 타일이 셋 이어지지 않는다.
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

fn chests(g: &Game) -> Vec<(i32, i32, u8)> {
    let mut v = Vec::new();
    for x in 1..=6 {
        for y in 1..=7 {
            if let Some(t) = g.grid[x][y] {
                if g.tiles[t].kind == Kind::Chest {
                    v.push((x as i32, y as i32, g.tiles[t].tier));
                }
            }
        }
    }
    v
}

#[test]
fn merge_three_normal_chests() {
    let mut game = Game::new_game(7, 11);
    game.load_board(BOARD);
    let mut env = Env::from_game(game);
    env.chest_bonus = [0.0, 0.1, 3.0, 10.0, 30.0];
    assert_eq!(env.game.phase, Phase::Idle);
    assert_eq!(chests(&env.game).len(), 3);
    assert_eq!(env.max_chest_tier(), 1, "시작 보드의 일반 상자는 이미 본 것으로 친다");

    // 패턴 검사: 교환 한 번이면 일반 상자 합성
    assert_eq!(Board::of(&env.game).one_swap_chest_merges(), 1 << 1);
    assert_eq!(Board::of(&env.game).chest_merge_distance(), Some(1));
    assert!(env.chest_merge_available());

    // 행동 마스크: (4,1) 왼쪽 드래그 = ((1-1)*6 + (4-1))*4 + 2(왼쪽) = 14
    let drag = 14;
    let mut mask = vec![false; N_ACTIONS];
    env.mask(&mut mask);
    assert!(mask[drag], "합성 드래그가 마스크에서 꺼져 있다");

    let out = env.step(drag).expect("합성 드래그가 무효");
    let g = &env.game;
    // 엔진: 동상자 하나, 일반 상자 없음
    let c = chests(g);
    assert_eq!(c.iter().filter(|c| c.2 == 2).count(), 1, "동상자가 하나 생겨야 한다: {c:?}");
    assert_eq!(c.iter().filter(|c| c.2 == 1).count(), 0, "일반 상자가 남았다: {c:?}");
    assert_eq!(g.chest_made[2], 1);
    assert_eq!(g.chest_made[1], 0, "시작 보드의 상자는 생성으로 세지 않는다");
    assert_eq!(env.max_chest_tier(), 2);
    // 보너스: 동상자 최초 생성 3.0 (생존·보스 변화 없음)
    assert!((out.reward - 3.0).abs() < 1e-6, "보상 {}", out.reward);
    assert!((env.ep_chest - 3.0).abs() < 1e-6);
    assert_eq!((env.merge_opps, env.merge_taken, env.max_normal_held), (1, 1, 3));

    // 관측: 동상자 칸의 종류(상자)와 등급(2) 채널
    let (bx, by, _) = *c.iter().find(|c| c.2 == 2).unwrap();
    let (mut grid, mut scal) = (vec![0.0f32; GRID_LEN], vec![0.0f32; N_SCALAR]);
    env.write_obs(&mut grid, &mut scal);
    assert_eq!(grid[grid_index(CH_KIND + kind_index(Kind::Chest).unwrap(), bx, by)], 1.0);
    assert_eq!(grid[grid_index(CH_TIER + 2, bx, by)], 1.0);
    assert_eq!(grid[grid_index(CH_TIER + 1, bx, by)], 0.0);

    // 개봉: 동상자 탭 → 스왑 +12, 개봉 카운터, 보너스는 다시 주지 않는다
    let swaps = env.game.swaps;
    env.mask(&mut mask);
    let tap = A_CELL + by as usize * 6 + (bx - 1) as usize;
    assert!(mask[tap], "동상자 탭이 마스크에서 꺼져 있다");
    let out = env.step(tap).expect("동상자 탭이 무효");
    assert_eq!(env.game.chest_opened[2], 1);
    if env.game.phase == Phase::Idle {
        assert!(env.game.swaps >= swaps + 12, "스왑 {} → {}", swaps, env.game.swaps);
    }
    assert!(out.reward.abs() < 1e-6, "개봉에 보너스가 붙었다: {}", out.reward);
}

/// 교환 두 번이면 합쳐지는 보드: 1행 h1 s0 h1 l0 h1 — (5,1)을 왼쪽으로 두 번 옮겨야 한다
#[test]
fn merge_distance_two() {
    let board = BOARD.replacen("h1h1s0h1l0i0", "h1s0h1l0h1i0", 1);
    let mut game = Game::new_game(7, 11);
    game.load_board(&board);
    let b = Board::of(&game);
    assert_eq!(b.one_swap_chest_merges(), 0);
    assert_eq!(b.chest_merge_distance(), Some(2));
    assert_eq!(b.chest_counts()[1], 3);
}
