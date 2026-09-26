//! 원본 대조용 CLI. oracle/trace.js와 같은 입출력 형식을 쓴다.
//! 입력(stdin, 한 줄에 한 케이스):
//!   {"vseed":u32,"mseed":u32,
//!    "board":"...",                      (선택) 원본 lR 형식 보드
//!    "setup":{"day":d,"hearts":h,"achievements":a},  (선택)
//!    "dayoff":true|false,                (선택) 휴식일 제안에 대한 자동 응답, 기본 false
//!    "actions":[{"drag":[x,y,d]}, ...]}
//! 스왑이 0이 되면 밤을 자동으로 진행한다(원본 기본 동작과 같음).
//! 출력(stdout, 한 줄에 한 케이스): {"states":[...]}
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
mod script;
use towerswap_core::{Dir, Game, Phase};

fn state_json(g: &Game) -> Value {
    json!({
        "tK": g.phase.code(),
        "day": g.day,
        "hearts": g.hearts,
        "swaps": g.swaps,
        "sN": g.achievements,
        "bossCol": g.boss_col,
        "dayOff": g.day_off_day,
        "deals": g.deals_done,
        "declined": g.deals_declined,
        "vseed": g.rng_v.seed,
        "mseed": g.rng_m.seed,
        "board": g.board_cells(),
        "turrets": g.turret_cells(),
        "pending": format!("{:?}", g.pending.devil.map(|d| d.1)),
    })
}

/// 입력 대기 상태가 될 때까지 자동 진행(밤, 휴식일 응답)
fn settle(g: &mut Game, dayoff: bool) {
    loop {
        match g.phase {
            Phase::Dusk => g.start_night(),
            Phase::DayOffOffer => g.answer_day_off(dayoff),
            _ => return,
        }
    }
}

fn run_case(c: &Value) -> Value {
    let vseed = c["vseed"].as_u64().unwrap() as u32;
    let mseed = c["mseed"].as_u64().unwrap() as u32;
    let dayoff = c["dayoff"].as_bool().unwrap_or(false);
    let mut g = Game::new_game(vseed, mseed);
    if let Some(board) = c["board"].as_str() {
        g.load_board(board);
    }
    if let Some(s) = c.get("setup") {
        if let Some(d) = s["day"].as_i64() {
            g.day = d as i32;
        }
        if let Some(h) = s["hearts"].as_i64() {
            g.hearts = h as i32;
        }
        if let Some(a) = s["achievements"].as_i64() {
            g.achievements = a as i32;
        }
    }
    let mut states = vec![state_json(&g)];
    if let Some(actions) = c["actions"].as_array() {
        for a in actions {
            if g.phase == Phase::GameOver {
                break;
            }
            if let Some(r) = a.get("r") {
                // 스크립트 모드: 밤은 Dusk에서 해석 규칙에 따라 시작된다
                script::interpret(&mut g, r.as_u64().unwrap());
                states.push(state_json(&g));
                continue;
            }
            if let Some(d) = a.get("drag") {
                let v: Vec<i64> = d.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                g.drag(v[0] as i32, v[1] as i32, Dir::from_index(v[2] as usize));
            } else {
                panic!("지원하지 않는 행동: {a}");
            }
            settle(&mut g, dayoff);
            states.push(state_json(&g));
        }
    }
    json!({ "states": states })
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let c: Value = serde_json::from_str(&line).unwrap();
        writeln!(out, "{}", run_case(&c)).unwrap();
    }
}
