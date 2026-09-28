//! 행동 마스크 검증: 무작위 롤아웃의 모든 상태에서 225개 행동을 복제본에 적용해 본다.
//! - 마스크가 켠 행동은 반드시 유효하다.
//! - 드래그 마스크는 실제 유효성과 정확히 같다(제한이 없는 평일).
//! - 입력을 받는 단계에는 유효한 행동이 하나 이상 있다.
//! - 휴식일 버리기 금지를 켠 게임(절반)에서는 휴식일의 버리기 드래그가 마스크와 step 모두에서 막힌다.
//! - 일반 상자 개봉 금지를 켠 게임(4판 중 1판 꼴)에서는 1등급 상자 탭이 마스크와 step 모두에서 막힌다.
use towerswap_core::env::{Env, A_CELL, A_YES, N_ACTIONS};
use towerswap_core::kinds::Kind;
use towerswap_core::{Dir, Game, Phase};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn int(&mut self, a: i64, b: i64) -> i64 {
        a + (self.next() % (b - a + 1) as u64) as i64
    }
    fn f(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

/// oracle/compare_items.js의 randomBoard와 같은 분포 (상자·아이템이 많은 보드)
fn item_board(r: &mut Lcg) -> String {
    let pick = |r: &mut Lcg, a: &str| a.chars().nth(r.int(0, a.len() as i64 - 1) as usize).unwrap();
    let mut s = String::new();
    for x in 1..=6 {
        s += &if (x == 1 || x == 6) && r.f() < 0.3 { format!("t{}", r.int(1, 3)) } else { "e0".into() };
    }
    for _ in 0..36 {
        let v = r.f();
        s += &if v < 0.40 {
            format!("{}0", pick(r, "sligd"))
        } else if v < 0.62 {
            format!("{}{}", pick(r, "tbcw"), r.int(1, 4))
        } else if v < 0.80 {
            format!("h{}", r.int(1, 4))
        } else if v < 0.85 {
            format!("a{}", r.int(1, 4))
        } else if v < 0.89 {
            format!("T{}", r.int(1, 3))
        } else if v < 0.93 {
            "F0".into()
        } else if v < 0.95 {
            "H2".into()
        } else if v < 0.97 {
            format!("{}{}", pick(r, "CD"), r.int(0, 3))
        } else {
            "e0".into()
        };
    }
    for _ in 0..6 {
        s += &if r.f() < 0.2 { format!("I{}", r.int(1, 3)) } else { "o0".into() };
    }
    s
}

#[test]
fn mask_matches_validity() {
    let mut r = Lcg(12345);
    let mut mask = vec![false; N_ACTIONS];
    let (mut states, mut episodes_done, mut phases) = (0u64, 0u64, std::collections::HashSet::new());
    let (mut toss_blocked, mut open_blocked) = (0u64, 0u64);
    for i in 0..300 {
        let mut game = Game::new_game(r.next() as u32 | 1, r.next() as u32 | 1);
        if i % 3 != 0 {
            game.load_board(&item_board(&mut r));
            let mut day = r.int(1, 45) as i32;
            if day % 10 == 0 {
                day += 1;
            }
            game.day = day;
            game.hearts = r.int(2, 36) as i32;
            game.achievements = ((day - 1) / 10).min(5);
        }
        let mut env = Env::from_game(game);
        env.no_toss_day_off = i % 2 == 1;
        env.no_open_normal = i % 4 < 2 && i % 3 != 0;
        for _ in 0..1500 {
            if env.done() {
                episodes_done += 1;
                break;
            }
            env.mask(&mut mask);
            let valid: Vec<usize> = (0..N_ACTIONS).filter(|&a| mask[a]).collect();
            assert!(!valid.is_empty(), "유효 행동 없음: {:?}", env.game.phase);
            phases.insert(format!("{:?}", env.game.phase));
            let weekday = env.game.phase == Phase::Idle && env.game.day != env.game.day_off_day;
            let off_ban = env.no_toss_day_off && env.game.phase == Phase::Idle && env.game.day == env.game.day_off_day;
            for a in 0..N_ACTIONS {
                let ok = env.clone().step(a).is_some();
                if mask[a] {
                    assert!(ok, "마스크가 켠 행동 {a}가 무효 ({:?})", env.game.phase);
                }
                if weekday && a < A_CELL {
                    assert_eq!(mask[a], ok, "드래그 {a} 마스크 불일치");
                }
                if off_ban && a < A_CELL {
                    let c = a / 4;
                    if env.game.drag_is_toss((c % 6) as i32 + 1, (c / 6) as i32 + 1, Dir::from_index(a % 4)) {
                        assert!(!mask[a] && !ok, "휴식일 버리기 {a}가 막히지 않음");
                        toss_blocked += 1;
                    }
                }
                let tap_phase = matches!(env.game.phase, Phase::Idle | Phase::Dusk);
                if env.no_open_normal && tap_phase && (A_CELL..A_YES).contains(&a) {
                    let c = a - A_CELL;
                    let (x, y) = ((c % 6) as i32 + 1, (c / 6) as i32);
                    if env.game.tile_at(x, y).map_or(false, |t| env.game.tiles[t].kind == Kind::Chest && env.game.tiles[t].tier <= 1) {
                        assert!(!mask[a] && !ok, "일반 상자 탭 {a}이 막히지 않음");
                        open_blocked += 1;
                    }
                }
            }
            states += 1;
            let a = valid[(r.next() as usize) % valid.len()];
            env.step(a).unwrap();
        }
    }
    eprintln!("상태 {states}개, 끝난 게임 {episodes_done}개, 단계 {phases:?}, 막힌 휴식일 버리기 {toss_blocked}개, 막힌 일반 상자 탭 {open_blocked}개");
    assert!(states > 10_000);
    assert!(toss_blocked > 0);
    assert!(open_blocked > 0);
}
