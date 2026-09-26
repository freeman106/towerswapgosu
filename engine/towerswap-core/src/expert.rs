//! greedy 전문가: 한 수 앞을 복제본으로 시험해 휴리스틱(타워 방어력 + 하트 + 스왑)이 가장 높은 행동을 고른다.
//! 기준선 평가와 모방 학습의 라벨로 쓴다.
use crate::env::{Env, A_CELL, A_END_DAY, A_NO, A_YES, N_ACTIONS};
use crate::game::{Game, Phase};
use crate::kinds::Kind;
use crate::rng::JsRng;

/// 간단한 난수 (동점 깨기용)
pub fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

/// 휴리스틱 파라미터. 환경변수 TS_EXPERT="swap=0.2,c1=6,..."로 바꿀 수 있다(조정 실험용).
struct Params {
    swap: f64,       // 스왑 1개의 가치
    chest: [f64; 5], // 상자 등급별 가치(스왑 단위). 기본은 개봉 스왑(2, 12, 70, 380): 스왑 + 상자 = 경제 가치
    treasure: f64,   // 보물 1개의 가치(스왑 단위)
    dusk_tier: u8,   // 밤 직전에 여는 상자의 최소 등급
    heart: f64,
}

fn params() -> &'static Params {
    static P: std::sync::OnceLock<Params> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut p = Params { swap: 0.2, chest: [0.0, 2.0, 12.0, 70.0, 380.0], treasure: 0.0, dusk_tier: 1, heart: 3.0 };
        for kv in std::env::var("TS_EXPERT").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("TS_EXPERT 형식: k=v,...");
            let v: f64 = v.parse().expect("TS_EXPERT 값");
            match k {
                "swap" => p.swap = v,
                "c1" | "c2" | "c3" | "c4" => p.chest[k[1..].parse::<usize>().unwrap()] = v,
                "treasure" => p.treasure = v,
                "dusk" => p.dusk_tier = v as u8,
                "heart" => p.heart = v,
                _ => panic!("TS_EXPERT: 모르는 키 {k}"),
            }
        }
        p
    })
}

/// 보드 평가: 타워 방어력 + 상자(스왑 환산) + 하트 + 스왑.
/// 정석(상자를 최대한 합친 뒤 열어 대량의 스왑으로 필드를 키운다)을 따르도록 상자를 합칠수록 가치가 커진다.
pub fn heuristic(g: &Game) -> f64 {
    if g.phase == Phase::GameOver {
        return -1e9;
    }
    let p = params();
    let w = |kind: Kind, tier: u8| -> f64 {
        let t = 3f64.powi(tier.max(1) as i32 - 1);
        match kind {
            Kind::Ballista | Kind::ArrowTower | Kind::Cannon => t,
            Kind::IceWall | Kind::Iceberg => 0.5 * t,
            Kind::Chest => p.swap * p.chest[tier.min(4) as usize],
            Kind::Treasure => p.swap * p.treasure,
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
    v + p.heart * g.hearts as f64 + p.swap * g.swaps.max(0) as f64
}

/// 전문가가 고려하지 않는 행동: 대포 방향 전환, 요정 사용 (효과가 한 수 앞에 드러나지 않는다)
fn skipped(g: &Game, a: usize) -> bool {
    if !(A_CELL..A_YES).contains(&a) || !matches!(g.phase, Phase::Idle | Phase::Dusk) {
        return false;
    }
    let c = a - A_CELL;
    g.grid[c % 6 + 1][c / 6].map_or(false, |t| matches!(g.tiles[t].kind, Kind::Cannon | Kind::Fairy | Kind::FairyHouse))
}

/// 행동 a 뒤의 평가. TNT 창을 여는 탭은 이어지는 최선의 폭파까지 본다.
/// 복제본의 난수는 `seeds`로 바꾼다: 실제 게임의 미래(떨어질 타일, 상자 내용물)를 보지 않는다.
/// 한 결정 안에서는 모든 후보가 같은 시드를 쓰므로 난수를 쓰지 않는 수끼리는 결과가 같다.
fn lookahead(env: &Env, a: usize, seeds: (u32, u32)) -> f64 {
    let mut e = env.clone();
    e.game.rng_v = JsRng::new(seeds.0);
    e.game.rng_m = JsRng::new(seeds.1);
    e.step(a).unwrap();
    if e.game.phase != Phase::TntMenu {
        return heuristic(&e.game);
    }
    let mut m = [false; N_ACTIONS];
    e.mask(&mut m);
    (0..N_ACTIONS)
        .filter(|&b| m[b])
        .map(|b| {
            let mut f = e.clone();
            f.step(b).unwrap();
            heuristic(&f.game)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// greedy의 최선 행동 집합(휴리스틱 동점 전부)을 `set`에 표시하고, 그중 하나를 무작위로 고른다.
/// `mask`는 env.mask()의 결과
pub fn greedy_set(env: &Env, mask: &[bool], set: &mut [bool], rng: &mut u64) -> usize {
    set.fill(false);
    let g = &env.game;
    let one = |set: &mut [bool], a: usize| {
        set[a] = true;
        a
    };
    match g.phase {
        Phase::Devil => {
            let price = g.pending.devil.unwrap().1;
            return one(set, if price == 0 || g.hearts - price >= 8 { A_YES } else { A_NO });
        }
        Phase::Shop => return one(set, A_YES),
        Phase::DayOffOffer => return one(set, A_NO),
        Phase::Dusk => {
            // 등급이 충분한 상자가 있으면 연다(그날 쓸 스왑), 없으면 밤
            for a in A_CELL..A_YES {
                let c = a - A_CELL;
                if mask[a]
                    && g.grid[c % 6 + 1][c / 6].map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier >= params().dusk_tier)
                {
                    set[a] = true;
                }
            }
            if !set.iter().any(|&x| x) {
                return one(set, A_END_DAY);
            }
        }
        _ => {
            let base = heuristic(g);
            let seeds = (lcg(rng) as u32 | 1, lcg(rng) as u32 | 1);
            let mut vals = [f64::NEG_INFINITY; N_ACTIONS];
            let mut best = f64::NEG_INFINITY;
            for a in 0..N_ACTIONS {
                if mask[a] && a != A_END_DAY && !skipped(g, a) {
                    vals[a] = lookahead(env, a, seeds);
                    best = best.max(vals[a]);
                }
            }
            // 휴식일에는 나아지는 행동이 없으면 하루를 끝낸다 (고려할 행동이 없어도)
            if mask[A_END_DAY] && best <= base + 1e-3 {
                return one(set, A_END_DAY);
            }
            for a in 0..N_ACTIONS {
                set[a] = vals[a] >= best - 1e-9;
            }
        }
    }
    let n = set.iter().filter(|&&x| x).count();
    let k = (lcg(rng) as usize) % n;
    (0..N_ACTIONS).filter(|&a| set[a]).nth(k).unwrap()
}

/// greedy 행동 (최선 행동 집합에서 무작위)
pub fn greedy_action(env: &Env, mask: &[bool], rng: &mut u64) -> usize {
    let mut set = [false; N_ACTIONS];
    greedy_set(env, mask, &mut set, rng)
}
