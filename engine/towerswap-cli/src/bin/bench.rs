//! 처리량 측정: 무작위 정책으로 게임을 돌려 초당 행동 수와 밤 비용을 잰다(단일 스레드).
//! 사용: bench [new|strong] [게임 수] [시작 시드]
//!   new    — 새 게임부터 무작위 유효 행동 (학습 초반 분포)
//!   strong — 타워가 많은 무작위 보드, 1~58일차에서 시작 (후반 밤 비용)
use std::time::{Duration, Instant};
use towerswap_core::kinds::Kind;
use towerswap_core::{ActionResult, Dir, Game, Phase};

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

const GENIE_KINDS: [Kind; 5] = [Kind::Wood, Kind::Stone, Kind::Iron, Kind::Treasure, Kind::IceCube];

fn cells(y0: i32) -> impl Iterator<Item = (i32, i32)> {
    (y0..=7).flat_map(|y| (1..=6).map(move |x| (x, y)))
}

/// compare_full.js의 randomBoard와 같은 분포
fn strong_board(r: &mut Lcg) -> String {
    let pick = |r: &mut Lcg, a: &[&str]| a[r.int(0, a.len() as i64 - 1) as usize].to_string();
    let mut s = String::new();
    for x in 1..=6 {
        if (x == 1 || x == 6) && r.f() < 0.5 {
            s += &format!("t{}", r.int(1, 4));
        } else {
            s += "e0";
        }
    }
    for _ in 0..36 {
        let v = r.f();
        s += &if v < 0.45 {
            pick(r, &["s", "l", "i", "g", "d"]) + "0"
        } else if v < 0.9 {
            pick(r, &["t", "b", "c", "w", "t", "b", "c"]) + &r.int(1, 4).to_string()
        } else if v < 0.95 {
            format!("f{}", r.int(1, 4))
        } else {
            pick(r, &["C", "D"]) + &r.int(1, 4).to_string()
        };
    }
    for _ in 0..6 {
        s += &if r.f() < 0.25 { format!("I{}", r.int(1, 3)) } else { "o0".to_string() };
    }
    s
}

#[derive(Default)]
struct Stats {
    actions: u64,  // 유효 행동
    attempts: u64, // 시도(무효 포함)
    nights: u64,
    night_time: Duration,
    games: u64,
    day_sum: u64,
    max_day: i32,
    night_by_day: Vec<(u64, Duration)>, // 10일 단위
}

/// 무작위 유효 행동 1개를 적용한다
fn random_step(g: &mut Game, r: &mut Lcg, st: &mut Stats) {
    let ok = |res: ActionResult| res == ActionResult::Ok;
    match g.phase {
        // 휴식일에는 평균 50행동 뒤 Done (보드가 비면 유효 행동이 없어진다)
        Phase::Idle if g.day == g.day_off_day && r.f() < 0.02 => {
            g.day_off_done();
        }
        Phase::Idle => loop {
            st.attempts += 1;
            if g.day == g.day_off_day && st.attempts % 1000 == 0 {
                g.day_off_done();
                break;
            }
            let (x, y) = (r.int(1, 6) as i32, r.int(1, 7) as i32);
            let res = if r.f() < 0.85 { g.drag(x, y, Dir::from_index(r.int(0, 3) as usize)) } else { g.tap(x, y) };
            if ok(res) {
                break;
            }
        },
        Phase::Dusk => {
            st.attempts += 1;
            if r.f() < 0.3 && ok(g.tap(r.int(1, 6) as i32, r.int(1, 7) as i32)) {
                // 무료 탭
            } else {
                let day = g.day;
                let t = Instant::now();
                g.start_night();
                let dt = t.elapsed();
                st.nights += 1;
                st.night_time += dt;
                let b = (day as usize - 1) / 10;
                if st.night_by_day.len() <= b {
                    st.night_by_day.resize(b + 1, (0, Duration::ZERO));
                }
                st.night_by_day[b].0 += 1;
                st.night_by_day[b].1 += dt;
            }
        }
        Phase::DayOffOffer => g.answer_day_off(r.f() < 0.5),
        Phase::Devil => {
            g.devil_answer(r.f() < 0.5);
        }
        Phase::Shop => {
            g.shop_answer(r.f() < 0.5);
        }
        Phase::Genie => {
            g.genie_pick(GENIE_KINDS[r.int(0, 4) as usize]);
        }
        Phase::Merchant => {
            let o: Vec<_> = cells(1).filter(|&(x, y)| g.merchant_can_sell(x, y)).collect();
            let (x, y) = o[r.int(0, o.len() as i64 - 1) as usize];
            g.merchant_sell(x, y);
        }
        Phase::PlaceItem => {
            let (k, t) = g.pending.place.unwrap();
            let o: Vec<_> = cells(0).filter(|&(x, y)| g.can_place_public(k, t, x, y)).collect();
            let (x, y) = o[r.int(0, o.len() as i64 - 1) as usize];
            g.place_item(x, y);
        }
        Phase::TntMenu => {
            let o: Vec<_> = cells(1).filter(|&(x, y)| g.tnt_can_target(x, y)).collect();
            let i = r.int(0, o.len() as i64) as usize;
            if i == 0 {
                g.tnt_blast_all();
            } else {
                g.tnt_blast_one(o[i - 1].0, o[i - 1].1);
            }
        }
        Phase::FairySource => {
            let o: Vec<_> = cells(1).filter(|&(x, y)| g.fairy_can_source(x, y)).collect();
            let (x, y) = o[r.int(0, o.len() as i64 - 1) as usize];
            g.fairy_pick_source(x, y);
        }
        Phase::FairyTarget => {
            let o: Vec<_> = cells(0).filter(|&(x, y)| g.fairy_can_target(x, y)).collect();
            if o.is_empty() {
                g.fairy_cancel();
            } else {
                let (x, y) = o[r.int(0, o.len() as i64 - 1) as usize];
                g.fairy_pick_target(x, y);
            }
        }
        p => panic!("입력 단계가 아님: {p:?}"),
    }
    st.actions += 1;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("new").to_string();
    let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut r = Lcg(seed);
    let mut st = Stats::default();
    let max_actions = 3000;
    let t0 = Instant::now();
    for _ in 0..n {
        let vseed = r.int(1, 2147483000) as u32;
        let mseed = r.int(1, 2147483000) as u32;
        let mut g = Game::new_game(vseed, mseed);
        if mode == "strong" {
            let board = strong_board(&mut r);
            g.load_board(&board);
            let mut day = r.int(1, 58) as i32;
            if day % 10 == 0 {
                day += 1;
            }
            g.day = day;
            g.hearts = r.int(3, 36) as i32;
            g.achievements = ((day - 1) / 10).min(5);
        }
        for _ in 0..max_actions {
            if g.phase == Phase::GameOver {
                break;
            }
            random_step(&mut g, &mut r, &mut st);
        }
        st.games += 1;
        st.day_sum += g.day as u64;
        st.max_day = st.max_day.max(g.day);
    }
    let total = t0.elapsed().as_secs_f64();
    let nt = st.night_time.as_secs_f64();
    println!("모드 {mode}: 게임 {} · 평균 도달 day {:.1} · 최대 day {}", st.games, st.day_sum as f64 / st.games as f64, st.max_day);
    println!("총 {:.2}s (밤 {:.2}s = {:.0}%)", total, nt, 100.0 * nt / total);
    println!(
        "유효 행동 {} ({:.0}/s, 시도 {:.1}회/행동) · 밤 {} ({:.0}/s, 평균 {:.3} ms)",
        st.actions,
        st.actions as f64 / total,
        st.attempts as f64 / st.actions as f64,
        st.nights,
        st.nights as f64 / total,
        1000.0 * nt / st.nights.max(1) as f64
    );
    println!("낮만: {:.0} 행동/s", (st.actions - st.nights) as f64 / (total - nt));
    for (b, (c, d)) in st.night_by_day.iter().enumerate() {
        if *c > 0 {
            println!("  day {:>2}~{:>2}: 밤 {:>6}회, 평균 {:.3} ms", b * 10 + 1, b * 10 + 10, c, 1000.0 * d.as_secs_f64() / *c as f64);
        }
    }
}
