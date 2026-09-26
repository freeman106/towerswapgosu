//! 원본 대조용 "스크립트" 모드: 행동 목록의 정수 r을 현재 상태에 따라 해석한다.
//! oracle/driver.js의 interpret()와 규칙이 같아야 한다.
use towerswap_core::kinds::Kind;
use towerswap_core::{Dir, Game, Phase};

const GENIE_KINDS: [Kind; 5] = [Kind::Wood, Kind::Stone, Kind::Iron, Kind::Treasure, Kind::IceCube];

fn cells(y0: i32) -> impl Iterator<Item = (i32, i32)> {
    (y0..=7).flat_map(|y| (1..=6).map(move |x| (x, y)))
}

/// r을 해석해 한 행동을 적용한다. 밤·휴식일 제안은 호출자가 처리한다.
pub fn interpret(g: &mut Game, r: u64) {
    let pick = |n: usize| (r as usize) % n;
    match g.phase {
        Phase::Idle => {
            let x = 1 + ((r / 7) % 6) as i32;
            let y = 1 + ((r / 41) % 7) as i32;
            if r % 100 < 75 {
                g.drag(x, y, Dir::from_index(((r / 293) % 4) as usize));
            } else {
                g.tap(x, y);
            }
        }
        Phase::Dusk => {
            if r % 3 != 0 {
                let x = 1 + ((r / 7) % 6) as i32;
                let y = 1 + ((r / 41) % 7) as i32;
                g.tap(x, y);
            } else {
                g.start_night();
            }
        }
        Phase::DayOffOffer => g.answer_day_off(r % 2 == 0),
        Phase::Devil => {
            g.devil_answer(r % 2 == 0);
        }
        Phase::Shop => {
            g.shop_answer(r % 2 == 0);
        }
        Phase::Genie => {
            g.genie_pick(GENIE_KINDS[pick(5)]);
        }
        Phase::Merchant => {
            let opts: Vec<_> = cells(1).filter(|&(x, y)| g.merchant_can_sell(x, y)).collect();
            let (x, y) = opts[pick(opts.len())];
            g.merchant_sell(x, y);
        }
        Phase::PlaceItem => {
            let (k, t) = g.pending.place.unwrap();
            let opts: Vec<_> = cells(0).filter(|&(x, y)| g.can_place_public(k, t, x, y)).collect();
            let (x, y) = opts[pick(opts.len())];
            g.place_item(x, y);
        }
        Phase::TntMenu => {
            let opts: Vec<_> = cells(1).filter(|&(x, y)| g.tnt_can_target(x, y)).collect();
            let i = pick(opts.len() + 1);
            if i == 0 {
                g.tnt_blast_all();
            } else {
                let (x, y) = opts[i - 1];
                g.tnt_blast_one(x, y);
            }
        }
        Phase::FairySource => {
            let opts: Vec<_> = cells(1).filter(|&(x, y)| g.fairy_can_source(x, y)).collect();
            let (x, y) = opts[pick(opts.len())];
            g.fairy_pick_source(x, y);
        }
        Phase::FairyTarget => {
            let opts: Vec<_> = cells(0).filter(|&(x, y)| g.fairy_can_target(x, y)).collect();
            if opts.is_empty() {
                g.fairy_cancel();
            } else {
                let (x, y) = opts[pick(opts.len())];
                g.fairy_pick_target(x, y);
            }
        }
        _ => {}
    }
}
