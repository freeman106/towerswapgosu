//! 기준선 정책 평가: random(무작위 유효 행동), greedy(한 수 앞 휴리스틱, towerswap_core::expert),
//! plan(상자가 걸린 결정에서 짧은 계획을 비교하는 휴리스틱, towerswap_core::planner).
//! 생존과 함께 상자 지표(은·금상자 최초 생성일과 성공률, 등급별 개봉, 상자를 모은 채 죽은 비율)를 낸다.
//! 사용: baseline [random|greedy|plan] [게임 수] [시드] [스레드 수]   (파라미터: 환경변수 TS_EXPERT, TS_PLAN)
use std::thread;
use towerswap_core::env::{Env, A_CELL, A_END_DAY, A_YES, N_ACTIONS};
use towerswap_core::expert::{greedy_action, greedy_set, lcg};
use towerswap_core::planner::{chest_decision, plan_action};
use towerswap_core::kinds::Kind;
use towerswap_core::{Game, Phase};

const OPEN_SWAPS: [i64; 5] = [0, 2, 12, 70, 380];

#[derive(Default, Clone)]
struct GameStats {
    score: i64,
    day: i32,
    first_tier: [Option<i32>; 5], // 등급별 상자 최초 생성일
    opened: [u32; 5],             // 등급별 개봉 수
    hoard_at_death: i64,          // 죽을 때 보드에 남은 상자의 개봉 스왑 합
    plan: [u64; 5],               // plan: 계획한 결정, greedy 최선 집합 밖의 선택, 그중 합성·개봉·밤 시작
}

fn chests(g: &Game) -> impl Iterator<Item = u8> + '_ {
    (1..=6).flat_map(move |x| (1..=7).filter_map(move |y| g.grid[x][y])).filter(|&t| g.tiles[t].kind == Kind::Chest).map(|t| g.tiles[t].tier.min(4))
}

fn play(policy: &str, seed: u64) -> GameStats {
    let mut r = seed;
    let mut env = Env::new(lcg(&mut r) as u32 | 1, lcg(&mut r) as u32 | 1);
    let mut m = vec![false; N_ACTIONS];
    let mut st = GameStats::default();
    for _ in 0..500_000 {
        if env.done() {
            break;
        }
        env.mask(&mut m);
        let a = if policy == "greedy" {
            greedy_action(&env, &m, &mut r)
        } else if policy == "plan" {
            plan_action(&env, &m, &mut r)
        } else {
            let valid: Vec<usize> = (0..N_ACTIONS).filter(|&a| m[a]).collect();
            valid[(lcg(&mut r) as usize) % valid.len()]
        };
        let g = &env.game;
        let differ = policy == "plan" && chest_decision(g) && {
            let (mut set, mut r2) = (vec![false; N_ACTIONS], r);
            greedy_set(&env, &m, &mut set, &mut r2);
            st.plan[0] += 1;
            !set[a]
        };
        let (made0, opened0) = (g.chest_made[2..].iter().sum::<u32>(), g.chest_opened.iter().sum::<u32>());
        if (A_CELL..A_YES).contains(&a) && matches!(g.phase, Phase::Idle | Phase::Dusk) {
            let c = a - A_CELL;
            if let Some(t) = g.grid[c % 6 + 1][c / 6].filter(|&t| g.tiles[t].kind == Kind::Chest) {
                st.opened[g.tiles[t].tier.min(4) as usize] += 1;
            }
        }
        env.step(a).unwrap();
        let g = &env.game;
        if differ {
            st.plan[1] += 1;
            st.plan[2] += (g.chest_made[2..].iter().sum::<u32>() > made0) as u64;
            st.plan[3] += (g.chest_opened.iter().sum::<u32>() > opened0) as u64;
            st.plan[4] += (a == A_END_DAY) as u64;
        }
        for tier in chests(g) {
            st.first_tier[tier as usize].get_or_insert(g.day);
        }
    }
    let g = &env.game;
    st.score = g.score();
    st.day = g.day;
    st.hoard_at_death = chests(g).map(|t| OPEN_SWAPS[t as usize]).sum();
    st
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let policy = args.get(1).cloned().unwrap_or("greedy".into());
    let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let threads = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(thread::available_parallelism().map_or(8, |n| n.get()) as u64);
    let res: Vec<GameStats> = thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                let policy = policy.clone();
                s.spawn(move || (0..n).filter(|i| i % threads == k).map(|i| play(&policy, seed * 1_000_003 + i)).collect::<Vec<_>>())
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let k = res.len() as f64;
    let mean = |f: &dyn Fn(&GameStats) -> f64| res.iter().map(f).sum::<f64>() / k;
    let mut days: Vec<i32> = res.iter().map(|x| x.day).collect();
    days.sort();
    println!(
        "{policy}: 게임 {} · 점수 평균 {:.1} · day 평균 {:.2} (중앙 {}, 최대 {}) · 보스 통과 평균 {:.2} · 50일 돌파 {:.1}%",
        res.len(),
        mean(&|x| x.score as f64),
        mean(&|x| x.day as f64),
        days[days.len() / 2],
        days[days.len() - 1],
        mean(&|x| (x.score / 1000) as f64),
        100.0 * mean(&|x| (x.day > 50) as u8 as f64)
    );
    let names = ["", "일반", "동", "은", "금"];
    let mut line = String::from("  상자 생성률(최초 day 평균):");
    for t in 1..=4 {
        let made: Vec<i32> = res.iter().filter_map(|x| x.first_tier[t]).collect();
        let d = if made.is_empty() { 0.0 } else { made.iter().sum::<i32>() as f64 / made.len() as f64 };
        line += &format!(" {} {:.0}%({:.1})", names[t], 100.0 * made.len() as f64 / k, d);
    }
    println!("{line}");
    let opened: Vec<String> = (1..=4).map(|t| format!("{} {:.2}", names[t], mean(&|x| x.opened[t] as f64))).collect();
    println!(
        "  게임당 개봉: {} · 개봉 스왑 {:.0} · 죽을 때 동상자 이상 가치(≥12스왑)를 남긴 비율 {:.0}%",
        opened.join(", "),
        mean(&|x| (1..=4).map(|t| x.opened[t] as i64 * OPEN_SWAPS[t]).sum::<i64>() as f64),
        100.0 * mean(&|x| (x.hoard_at_death >= 12) as u8 as f64)
    );
    if policy == "plan" {
        let p: Vec<f64> = (0..5).map(|i| mean(&|x| x.plan[i] as f64)).collect();
        println!(
            "  게임당 계획한 결정 {:.1} · greedy와 다른 선택 {:.1} (합성 {:.2}, 개봉 {:.2}, 밤 시작 {:.2})",
            p[0], p[1], p[2], p[3], p[4]
        );
    }
}
