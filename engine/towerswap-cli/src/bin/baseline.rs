//! 기준선 정책 평가: random(무작위 유효 행동), greedy(한 수 앞 휴리스틱).
//! 사용: baseline [random|greedy] [게임 수] [시드]
use std::thread;
use towerswap_core::env::{Env, A_CELL, A_END_DAY, A_NO, A_YES, N_ACTIONS};
use towerswap_core::kinds::Kind;
use towerswap_core::{Game, Phase};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// 보드 평가: 타워 방어력 + 하트 + 스왑
fn heuristic(g: &Game) -> f64 {
    if g.phase == Phase::GameOver {
        return -1e9;
    }
    let w = |kind: Kind, tier: u8| -> f64 {
        let p = 3f64.powi(tier.max(1) as i32 - 1);
        match kind {
            Kind::Ballista | Kind::ArrowTower | Kind::Cannon => p,
            Kind::IceWall | Kind::Iceberg => 0.5 * p,
            Kind::Chest => 0.3 * p,
            _ => 0.0,
        }
    };
    let mut v = 0.0;
    for x in 1..=6 {
        for y in 1..=7 {
            if let Some(t) = g.grid[x][y] {
                v += w(g.tiles[t].kind, g.tiles[t].tier);
            }
        }
        if let Some(t) = g.turrets[x] {
            v += w(g.tiles[t].kind, g.tiles[t].tier);
        }
    }
    v + 1.5 * g.hearts as f64 + 0.2 * g.swaps.min(20) as f64
}

fn greedy(env: &Env, valid: &[usize], r: &mut Lcg) -> usize {
    let g = &env.game;
    match g.phase {
        Phase::Devil => {
            let price = g.pending.devil.unwrap().1;
            return if price == 0 || g.hearts - price >= 8 { A_YES } else { A_NO };
        }
        Phase::Shop => return A_YES,
        Phase::DayOffOffer => return A_NO,
        Phase::Dusk => {
            // 상자가 있으면 연다(무료 스왑), 없으면 밤
            for &a in valid {
                if a >= A_CELL && a < A_YES {
                    let c = a - A_CELL;
                    let t = g.grid[c % 6 + 1][c / 6];
                    if t.map_or(false, |t| g.tiles[t].kind == Kind::Chest) {
                        return a;
                    }
                }
            }
            return A_END_DAY;
        }
        _ => {}
    }
    let base = heuristic(g);
    let mut best = (f64::NEG_INFINITY, A_END_DAY);
    for &a in valid {
        if a == A_END_DAY {
            continue;
        }
        let mut e = env.clone();
        e.step(a).unwrap();
        let v = heuristic(&e.game) + (r.next() % 1000) as f64 * 1e-6;
        if v > best.0 {
            best = (v, a);
        }
    }
    // 휴식일에는 나아지는 행동이 없으면 하루를 끝낸다
    if valid.contains(&A_END_DAY) && best.0 <= base + 1e-3 {
        return A_END_DAY;
    }
    best.1
}

fn play(policy: &str, seed: u64) -> (i64, i32) {
    let mut r = Lcg(seed);
    let mut env = Env::new(r.next() as u32 | 1, r.next() as u32 | 1);
    let mut m = vec![false; N_ACTIONS];
    for _ in 0..200_000 {
        if env.done() {
            break;
        }
        env.mask(&mut m);
        let valid: Vec<usize> = (0..N_ACTIONS).filter(|&a| m[a]).collect();
        let a = if policy == "greedy" { greedy(&env, &valid, &mut r) } else { valid[(r.next() as usize) % valid.len()] };
        env.step(a).unwrap();
    }
    (env.game.score(), env.game.day)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let policy = args.get(1).cloned().unwrap_or("greedy".into());
    let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let threads = thread::available_parallelism().map_or(8, |n| n.get()) as u64;
    let res: Vec<(i64, i32)> = thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                let policy = policy.clone();
                s.spawn(move || (0..n).filter(|i| i % threads == k).map(|i| play(&policy, seed * 1_000_003 + i)).collect::<Vec<_>>())
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mean = |f: &dyn Fn(&(i64, i32)) -> f64| res.iter().map(f).sum::<f64>() / res.len() as f64;
    let mut days: Vec<i32> = res.iter().map(|x| x.1).collect();
    days.sort();
    println!(
        "{policy}: 게임 {} · 점수 평균 {:.1} · day 평균 {:.2} (중앙 {}, 최대 {}) · 보스 통과 평균 {:.2}",
        res.len(),
        mean(&|x| x.0 as f64),
        mean(&|x| x.1 as f64),
        days[days.len() / 2],
        days[days.len() - 1],
        mean(&|x| (x.0 / 1000) as f64)
    );
}
