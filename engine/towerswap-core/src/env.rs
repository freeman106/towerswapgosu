//! 강화학습 환경 계층: 행동 인코딩, 단계별 행동 마스크, 관측, 보상, 환경 수준 제한.
//!
//! 행동 번호 (N_ACTIONS = 225)
//! - 0..168   드래그: ((y-1)*6 + (x-1))*4 + 방향(위·아래·왼·오)
//! - 168..216 칸 선택: 168 + y*6 + (x-1), y = 0..7. 평소에는 탭, 선택 창에서는 대상 선택
//! - 216/217  예/아니오: 휴식일 제안, 악마 거래, 상점 무료 품목. 아니오는 요정 대상이 없을 때 요정 취소도 한다
//! - 218..223 지니 자원 종류 (GENIE_KINDS 순서)
//! - 223      TNT 전부 폭파
//! - 224      하루 끝내기: 밤 직전(Dusk)이면 밤 시작, 휴식일이면 Done 후 밤 시작
//!
//! 환경 수준 제한
//! - 대포 방향 전환은 대포마다 하루 1회
//! - 휴식일 행동은 DAY_OFF_CAP개까지 (넘으면 하루 끝내기만 가능)
//! - 선택: 휴식일 버리기 금지(`no_toss_day_off`, 기본 끔). 휴식일에는 합칠 것을 다 합칠 수 있어 버리기는 손해뿐이다
//! - 선택: 상자 개봉 규칙(`open_rule`, 기본 제한 없음). `min_tier` 미만 등급은 열 수 없고, 비상 예외로 지정한 등급만
//!   남은 스왑·하트가 기준 이하이고 지금 열 수 있는 `min_tier` 이상 상자가 없을 때 열 수 있다(허용일 뿐 강제 개봉은 아니다)
//! - TNT·요정 창의 취소는 막는다(상태가 그대로인 행동의 반복을 막는다). 요정은 가능한 이동이 있을 때만 쓸 수 있다.
//!
//! 보상 = 앱 점수 증가분 / 100 (하루 생존 +0.01, 보스 통과 +10). 한 스텝에 지난 날 수를 `days`로 알려 준다(하루 단위 할인용).
//! 선택 보상: 게임에서 각 등급 상자가 보드에 처음 나타나면 `chest_bonus[등급]`을 더한다(기본 0).
//! 선택 경제 조형(퍼텐셜 기반): Φ(s) = econ_w × (남은 스왑 + 보드 상자의 개봉 스왑), 보상에 γ^days·Φ(s′) − Φ(s)를 더한다.
//! 게임 오버 뒤 Φ = 0. γ(`econ_gamma`)는 학습의 하루 할인과 같아야 최적 정책이 바뀌지 않는다(기본 econ_w = 0).

use crate::analysis::Board;
use crate::game::{ActionResult, Dir, Game, Phase, TileId};
use crate::items::{chest_swaps, DevilOffer, ShopItem};
use crate::kinds::Kind;
use crate::level::{Terrain, COLS, ROWS};

pub const N_ACTIONS: usize = 225;
pub const A_DRAG: usize = 0;
pub const A_CELL: usize = 168;
pub const A_YES: usize = 216;
pub const A_NO: usize = 217;
pub const A_GENIE: usize = 218;
pub const A_TNT_ALL: usize = 223;
pub const A_END_DAY: usize = 224;
pub const DAY_OFF_CAP: u32 = 200;
pub const GENIE_KINDS: [Kind; 5] = [Kind::Wood, Kind::Stone, Kind::Iron, Kind::Treasure, Kind::IceCube];

// ───────────────────────── 관측 배치 ─────────────────────────

/// 격자: 0..7행 × 1..6열, 채널 우선 [채널][행][열]
pub const GRID_H: usize = ROWS + 1;
pub const GRID_W: usize = COLS;
const N_KIND: usize = 15;
pub const CH_KIND: usize = 0; // 종류 원-핫 15
pub const CH_TIER: usize = 15; // 등급 원-핫 0..4
const CH_WATER: usize = 20;
const CH_GROUND: usize = 21;
const CH_SLOT: usize = 22;
const CH_CASTLE: usize = 23; // 0행
const CH_TNT: usize = 24; // TNT 충전 / 3
const CH_LEFT: usize = 25; // 대포 선호 방향이 왼쪽
const CH_FLIPPED: usize = 26; // 오늘 이미 방향을 바꾼 대포
const CH_FAIRY_CHARGE: usize = 27; // 요정의 집 남은 사용 / 4
const CH_OPENED: usize = 28; // 여는 중인 상자 (상인에게 팔 수 없음)
const CH_TNT_SRC: usize = 29; // 창을 연 TNT
const CH_FAIRY: usize = 30; // 사용 중인 요정
const CH_FAIRY_SRC: usize = 31; // 요정이 집은 타일
const CH_DEVIL: usize = 32; // 악마 제안 대상 칸
const CH_BOSS: usize = 33; // 보스 열 (보스 날)
pub const N_GRID_CH: usize = 34;
pub const GRID_LEN: usize = N_GRID_CH * GRID_H * GRID_W;

const S_PHASE: usize = 14; // 단계 원-핫 11
const S_DEVIL_TYPE: usize = 25; // 4
const S_DEVIL_PRICE: usize = 29;
const S_DEVIL_KIND: usize = 30; // 15
const S_DEVIL_TIER: usize = 45;
const S_SHOP_TYPE: usize = 46; // 3
const S_SHOP_KIND: usize = 49; // 15
const S_SHOP_TIER: usize = 64;
const S_SHOP_HEARTS: usize = 65;
const S_PLACE_KIND: usize = 66; // 15
const S_PLACE_TIER: usize = 81;
pub const N_SCALAR: usize = 82;

const INPUT_PHASES: [Phase; 11] = [
    Phase::Idle,
    Phase::Dusk,
    Phase::DayOffOffer,
    Phase::Devil,
    Phase::Merchant,
    Phase::Shop,
    Phase::PlaceItem,
    Phase::Genie,
    Phase::TntMenu,
    Phase::FairySource,
    Phase::FairyTarget,
];

pub fn kind_index(k: Kind) -> Option<usize> {
    Some(match k {
        Kind::Anvil => 0,
        Kind::Tnt => 1,
        Kind::Iceberg => 2,
        Kind::Fairy => 3,
        Kind::FairyHouse => 4,
        Kind::Wood => 5,
        Kind::Ballista => 6,
        Kind::Stone => 7,
        Kind::ArrowTower => 8,
        Kind::Iron => 9,
        Kind::Cannon => 10,
        Kind::Treasure => 11,
        Kind::Chest => 12,
        Kind::IceCube => 13,
        Kind::IceWall => 14,
        Kind::CannonSlot | Kind::TimeMachine | Kind::You => return None,
    })
}

#[inline]
fn cell_action(x: i32, y: i32) -> usize {
    A_CELL + y as usize * COLS + (x - 1) as usize
}

#[inline]
pub fn grid_index(ch: usize, x: i32, y: i32) -> usize {
    (ch * GRID_H + y as usize) * GRID_W + (x - 1) as usize
}

/// 경제 가치(스왑 단위): 남은 스왑 + 보드 상자의 개봉 스왑
fn econ_units(swaps: i64, n: &[u32; 5]) -> f32 {
    (swaps.max(0) + (1..=4).map(|t| n[t] as i64 * chest_swaps(t as u8)).sum::<i64>()) as f32
}

/// 상자 개봉 규칙. 기본은 제한 없음(min_tier 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenRule {
    /// 평소 열 수 있는 최소 등급
    pub min_tier: u8,
    /// 비상 예외 (등급, 남은 스왑 상한, 하트 상한): 그 등급 상자는 스왑·하트가 상한 이하이고
    /// 지금 열 수 있는 min_tier 이상 상자가 없을 때만 열 수 있다
    pub emergency: Option<(u8, i64, i32)>,
}

impl Default for OpenRule {
    fn default() -> Self {
        OpenRule { min_tier: 1, emergency: None }
    }
}

pub struct StepOut {
    pub reward: f32,
    pub done: bool,
    /// 이 스텝에 지난 날 수 (밤을 넘기면 1)
    pub days: u32,
}

#[derive(Clone)]
pub struct Env {
    pub game: Game,
    flipped_today: Vec<TileId>, // 오늘 방향을 바꾼 대포
    day_actions: u32,           // 오늘 한 행동 수
    day: i32,                   // 위 두 기록의 날
    /// 등급별 상자 최초 생성 보상 (인덱스 = 등급 1..4)
    pub chest_bonus: [f32; 5],
    /// 경제 조형의 가중치(스왑 1개당)와 하루 할인
    pub econ_w: f32,
    pub econ_gamma: f32,
    /// 휴식일 버리기 금지 (마스크와 step 모두)
    pub no_toss_day_off: bool,
    /// 상자 개봉 규칙 (마스크와 step 모두)
    pub open_rule: OpenRule,
    /// 통계: 비상 예외로 연 상자 수
    pub emergency_opens: u32,
    chest_seen: u8, // 이 게임에서 보드에 나타난 상자 등급 (비트)
    /// 이 게임의 보상 성분별 합계: 생존(하루 0.01), 보스(10), 상자 최초 생성 보너스
    pub ep_survival: f32,
    pub ep_boss: f32,
    pub ep_chest: f32,
    pub ep_econ: f32,
    /// 통계: 교환 한 번으로 상자 합성이 가능했던 결정 수(입력 대기 상태), 그때 실제로 합성한 수,
    /// 동시에 보유한 일반 상자의 최대 수
    pub merge_opps: u32,
    pub merge_taken: u32,
    pub max_normal_held: u32,
    /// 통계: 이 에피소드의 첫 상자 합성 시점 (시작부터 센 행동 수, 시작일부터 지난 날)
    pub first_merge: Option<(u32, i32)>,
    /// 통계: 같은 등급 상자(1~3등급)를 2개 이상·3개 이상 동시에 가진 행동 수, 최대 동시 보유 수
    pub steps_hold2: u32,
    pub steps_hold3: u32,
    pub max_same_held: u32,
    start_day: i32,
    ep_steps: u32,
}

impl Env {
    pub fn new(vseed: u32, mseed: u32) -> Env {
        Env::from_game(Game::new_game(vseed, mseed))
    }

    /// 임의의 게임 상태에서 시작한다 (시나리오·검사용)
    pub fn from_game(game: Game) -> Env {
        let day = game.day;
        let mut env = Env {
            game,
            flipped_today: Vec::new(),
            day_actions: 0,
            day,
            chest_bonus: [0.0; 5],
            econ_w: 0.0,
            econ_gamma: 1.0,
            no_toss_day_off: false,
            open_rule: OpenRule::default(),
            emergency_opens: 0,
            chest_seen: 0,
            ep_survival: 0.0,
            ep_boss: 0.0,
            ep_chest: 0.0,
            ep_econ: 0.0,
            merge_opps: 0,
            merge_taken: 0,
            max_normal_held: 0,
            first_merge: None,
            steps_hold2: 0,
            steps_hold3: 0,
            max_same_held: 0,
            start_day: day,
            ep_steps: 0,
        };
        let (bits, n) = env.chest_census();
        env.chest_seen = bits;
        env.max_normal_held = n[1];
        env.max_same_held = n[1..4].iter().copied().max().unwrap_or(0);
        env
    }

    /// 보드에 있는 상자: (등급 비트 1 << 등급, 등급별 개수)
    fn chest_census(&self) -> (u8, [u32; 5]) {
        let g = &self.game;
        let (mut bits, mut n) = (0u8, [0u32; 5]);
        for col in &g.grid[1..] {
            for &t in col[1..].iter().flatten() {
                if g.tiles[t].kind == Kind::Chest {
                    let tier = g.tiles[t].tier.min(4);
                    bits |= 1 << tier;
                    n[tier as usize] += 1;
                }
            }
        }
        (bits, n)
    }

    /// 교환 한 번으로 상자 합성이 가능한지 (같은 등급 상자가 3개 이상일 때만 패턴 검사)
    pub fn chest_merge_available(&self) -> bool {
        let (_, n) = self.chest_census();
        n[1..4].iter().any(|&c| c >= 3) && Board::of(&self.game).one_swap_chest_merges() != 0
    }

    /// 이 에피소드를 시작한 날
    pub fn start_day(&self) -> i32 {
        self.start_day
    }

    /// 상자 t를 개봉 규칙상 열 수 있는지 (원래 게임 규칙의 가능 여부는 따로 본다)
    pub fn chest_rule_ok(&self, t: TileId) -> bool {
        let (g, r) = (&self.game, &self.open_rule);
        let tier = g.tiles[t].tier;
        if tier >= r.min_tier {
            return true;
        }
        match r.emergency {
            Some((et, max_swaps, max_hearts)) if tier == et => {
                g.swaps <= max_swaps && g.hearts <= max_hearts && !self.openable_chest_at_least(r.min_tier)
            }
            _ => false,
        }
    }

    /// 지금 원래 게임 규칙으로 열 수 있는 `min` 등급 이상 상자가 보드에 있는지
    fn openable_chest_at_least(&self, min: u8) -> bool {
        let g = &self.game;
        (1..=ROWS as i32).any(|y| {
            (1..=COLS as i32).any(|x| {
                g.tile_at(x, y).map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier >= min) && g.tap_valid(x, y)
            })
        })
    }

    /// 이 게임에서 만든 가장 높은 상자 등급 (없으면 0)
    pub fn max_chest_tier(&self) -> u8 {
        (1..=4).rev().find(|&t| self.chest_seen & (1 << t) != 0).unwrap_or(0)
    }

    pub fn done(&self) -> bool {
        self.game.phase == Phase::GameOver
    }

    // ───────────────────────── 마스크 ─────────────────────────

    fn tap_ok(&self, x: i32, y: i32, fairy_ok: &mut Option<bool>) -> bool {
        let g = &self.game;
        if !g.tap_valid(x, y) {
            return false;
        }
        let t = g.tile_at(x, y).unwrap();
        match g.tiles[t].kind {
            Kind::Cannon => !self.flipped_today.contains(&t),
            Kind::Chest => self.chest_rule_ok(t),
            Kind::Fairy | Kind::FairyHouse => *fairy_ok.get_or_insert_with(|| g.fairy_any_move()),
            _ => true,
        }
    }

    /// 유효한 행동 표시. 입력을 받는 단계라면 항상 하나 이상이다.
    pub fn mask(&self, m: &mut [bool]) {
        m.fill(false);
        let g = &self.game;
        let cells = |y0: i32| (y0..=ROWS as i32).flat_map(|y| (1..=COLS as i32).map(move |x| (x, y)));
        match g.phase {
            Phase::Idle | Phase::Dusk => {
                let closed = g.closed_day();
                if g.phase == Phase::Idle {
                    if closed && self.day_actions >= DAY_OFF_CAP {
                        m[A_END_DAY] = true;
                        return;
                    }
                    for (x, y) in cells(1) {
                        for d in 0..4 {
                            let a = A_DRAG + ((y - 1) as usize * COLS + (x - 1) as usize) * 4 + d;
                            let dir = Dir::from_index(d);
                            m[a] = g.drag_valid(x, y, dir) && !(closed && self.no_toss_day_off && g.drag_is_toss(x, y, dir));
                        }
                    }
                }
                let mut fairy_ok = None;
                for (x, y) in cells(1) {
                    m[cell_action(x, y)] = self.tap_ok(x, y, &mut fairy_ok);
                }
                m[A_END_DAY] = g.phase == Phase::Dusk || closed;
            }
            Phase::DayOffOffer | Phase::Devil | Phase::Shop => {
                m[A_YES] = true;
                m[A_NO] = true;
            }
            Phase::Merchant => {
                for (x, y) in cells(1) {
                    m[cell_action(x, y)] = g.merchant_can_sell(x, y);
                }
            }
            Phase::PlaceItem => {
                let (k, t) = g.pending.place.unwrap();
                for (x, y) in cells(0) {
                    m[cell_action(x, y)] = g.can_place_public(k, t, x, y);
                }
            }
            Phase::Genie => m[A_GENIE..A_GENIE + GENIE_KINDS.len()].fill(true),
            Phase::TntMenu => {
                m[A_TNT_ALL] = true;
                for (x, y) in cells(1) {
                    m[cell_action(x, y)] = g.tnt_can_target(x, y);
                }
            }
            Phase::FairySource => {
                let mut any = false;
                for (x, y) in cells(1) {
                    let ok = g.fairy_can_source(x, y) && g.fairy_source_has_target(g.tile_at(x, y).unwrap());
                    m[cell_action(x, y)] = ok;
                    any |= ok;
                }
                m[A_NO] = !any;
            }
            Phase::FairyTarget => {
                let mut any = false;
                for (x, y) in cells(0) {
                    let ok = g.fairy_can_target(x, y);
                    m[cell_action(x, y)] = ok;
                    any |= ok;
                }
                m[A_NO] = !any;
            }
            _ => {}
        }
    }

    // ───────────────────────── 스텝 ─────────────────────────

    /// 행동 하나를 적용한다. 무효한 행동은 상태를 바꾸지 않고 `None`을 돌려준다.
    pub fn step(&mut self, a: usize) -> Option<StepOut> {
        let (s0, d0, a0) = (self.game.score(), self.game.day, self.game.achievements);
        let phi0 = if self.econ_w != 0.0 { self.econ_w * econ_units(self.game.swaps, &self.chest_census().1) } else { 0.0 };
        let opp = self.game.phase == Phase::Idle && self.chest_merge_available();
        let merged0: u32 = self.game.chest_made[2..].iter().sum();
        self.game.stat_steps += 1;
        if !self.apply(a) {
            self.game.stat_steps -= 1;
            return None;
        }
        self.ep_steps += 1;
        let merged = self.game.chest_made[2..].iter().sum::<u32>() > merged0;
        if opp {
            self.merge_opps += 1;
            self.merge_taken += merged as u32;
        }
        if merged && self.first_merge.is_none() {
            self.first_merge = Some((self.ep_steps, self.game.day - self.start_day));
        }
        self.day_actions += 1;
        let (bits, n) = self.chest_census();
        self.max_normal_held = self.max_normal_held.max(n[1]);
        let same = n[1..4].iter().copied().max().unwrap_or(0);
        self.max_same_held = self.max_same_held.max(same);
        self.steps_hold2 += (same >= 2) as u32;
        self.steps_hold3 += (same >= 3) as u32;
        let new = bits & !self.chest_seen;
        self.chest_seen |= new;
        let bonus = (1..=4).filter(|&t| new & (1 << t) != 0).map(|t| self.chest_bonus[t]).sum::<f32>();
        self.ep_chest += bonus;
        self.ep_survival += (self.game.day - d0) as f32 * 0.01;
        self.ep_boss += (self.game.achievements - a0) as f32 * 10.0;
        let days = (self.game.day - d0) as u32;
        let econ = if self.econ_w != 0.0 {
            let phi1 = if self.game.phase == Phase::GameOver { 0.0 } else { self.econ_w * econ_units(self.game.swaps, &n) };
            self.econ_gamma.powi(days as i32) * phi1 - phi0
        } else {
            0.0
        };
        self.ep_econ += econ;
        let g = &self.game;
        if g.day != self.day {
            self.day = g.day;
            self.day_actions = 0;
            self.flipped_today.clear();
        }
        Some(StepOut {
            reward: (g.score() - s0) as f32 / 100.0 + bonus + econ,
            done: g.phase == Phase::GameOver,
            days,
        })
    }

    fn apply(&mut self, a: usize) -> bool {
        // 상자 개봉 규칙: 낮의 상자 탭만 해당 (허용되지 않으면 무효, 비상 예외로 열면 센다)
        let mut emergency = false;
        if (A_CELL..A_YES).contains(&a) && matches!(self.game.phase, Phase::Idle | Phase::Dusk) {
            let c = a - A_CELL;
            let (x, y) = ((c % COLS) as i32 + 1, (c / COLS) as i32);
            if let Some(t) = self.game.tile_at(x, y).filter(|&t| self.game.tiles[t].kind == Kind::Chest) {
                if !self.chest_rule_ok(t) {
                    return false;
                }
                emergency = self.game.tiles[t].tier < self.open_rule.min_tier;
            }
        }
        let g = &mut self.game;
        let r = if a < A_CELL {
            let c = (a - A_DRAG) / 4;
            let (x, y, dir) = ((c % COLS) as i32 + 1, (c / COLS) as i32 + 1, Dir::from_index(a % 4));
            if self.no_toss_day_off && g.closed_day() && g.drag_is_toss(x, y, dir) {
                return false;
            }
            g.drag(x, y, dir)
        } else if a < A_YES {
            let c = a - A_CELL;
            let (x, y) = ((c % COLS) as i32 + 1, (c / COLS) as i32);
            match g.phase {
                Phase::Idle | Phase::Dusk => {
                    let cannon = g.tile_at(x, y).filter(|&t| g.tiles[t].kind == Kind::Cannon);
                    if cannon.map_or(false, |t| self.flipped_today.contains(&t)) {
                        return false;
                    }
                    let r = g.tap(x, y);
                    if r == ActionResult::Ok {
                        self.flipped_today.extend(cannon);
                        self.emergency_opens += emergency as u32;
                    }
                    r
                }
                Phase::Merchant => g.merchant_sell(x, y),
                Phase::PlaceItem => g.place_item(x, y),
                Phase::TntMenu => g.tnt_blast_one(x, y),
                Phase::FairySource => g.fairy_pick_source(x, y),
                Phase::FairyTarget => g.fairy_pick_target(x, y),
                _ => ActionResult::Invalid,
            }
        } else if a == A_YES || a == A_NO {
            let yes = a == A_YES;
            match g.phase {
                Phase::DayOffOffer => {
                    g.answer_day_off(yes);
                    ActionResult::Ok
                }
                Phase::Devil => g.devil_answer(yes),
                Phase::Shop => g.shop_answer(yes),
                Phase::FairySource | Phase::FairyTarget if !yes => g.fairy_cancel(),
                _ => ActionResult::Invalid,
            }
        } else if a < A_TNT_ALL {
            g.genie_pick(GENIE_KINDS[a - A_GENIE])
        } else if a == A_TNT_ALL {
            g.tnt_blast_all()
        } else if a == A_END_DAY {
            if g.phase == Phase::Idle && g.day_off_done() == ActionResult::Invalid {
                return false;
            }
            if g.phase != Phase::Dusk {
                return false;
            }
            g.start_night();
            ActionResult::Ok
        } else {
            ActionResult::Invalid
        };
        r == ActionResult::Ok
    }

    // ───────────────────────── 관측 ─────────────────────────

    fn tile_features(&self, grid: &mut [f32], t: TileId, x: i32, y: i32) {
        let tile = &self.game.tiles[t];
        if let Some(k) = kind_index(tile.kind) {
            grid[grid_index(CH_KIND + k, x, y)] = 1.0;
        }
        grid[grid_index(CH_TIER + (tile.tier as usize).min(4), x, y)] = 1.0;
        match tile.kind {
            Kind::Tnt => grid[grid_index(CH_TNT, x, y)] = tile.dynamite as f32 / 3.0,
            Kind::Cannon => {
                grid[grid_index(CH_LEFT, x, y)] = tile.flipped_preferred as u8 as f32;
                grid[grid_index(CH_FLIPPED, x, y)] = self.flipped_today.contains(&t) as u8 as f32;
            }
            Kind::FairyHouse => grid[grid_index(CH_FAIRY_CHARGE, x, y)] = (tile.frame - 1).max(0) as f32 / 4.0,
            _ => {}
        }
    }

    /// 관측을 쓴다. grid: GRID_LEN, scal: N_SCALAR
    pub fn write_obs(&self, grid: &mut [f32], scal: &mut [f32]) {
        grid.fill(0.0);
        scal.fill(0.0);
        let g = &self.game;
        let mark = |grid: &mut [f32], ch: usize, t: Option<TileId>| {
            if let Some(t) = t {
                let (x, y) = (g.tiles[t].gx, g.tiles[t].gy);
                if (1..=COLS as i32).contains(&x) && (0..=ROWS as i32).contains(&y) {
                    grid[grid_index(ch, x, y)] = 1.0;
                }
            }
        };
        for x in 1..=COLS as i32 {
            grid[grid_index(CH_CASTLE, x, 0)] = 1.0;
            for y in 1..=ROWS as i32 {
                match g.terrain[x as usize][y as usize] {
                    Terrain::Water => grid[grid_index(CH_WATER, x, y)] = 1.0,
                    Terrain::Grass | Terrain::Road => grid[grid_index(CH_GROUND, x, y)] = 1.0,
                    Terrain::CannonSlot => {
                        grid[grid_index(CH_GROUND, x, y)] = 1.0;
                        grid[grid_index(CH_SLOT, x, y)] = 1.0;
                    }
                }
                if let Some(t) = g.grid[x as usize][y as usize] {
                    self.tile_features(grid, t, x, y);
                }
            }
            if let Some(t) = g.turrets[x as usize] {
                self.tile_features(grid, t, x, 0);
            }
            if g.day == g.goal_next() && g.boss_col == x {
                for y in 0..=ROWS as i32 {
                    grid[grid_index(CH_BOSS, x, y)] = 1.0;
                }
            }
        }
        mark(grid, CH_OPENED, g.opened_chest);
        mark(grid, CH_TNT_SRC, g.pending.tnt_source);
        mark(grid, CH_FAIRY, g.pending.fairy);
        mark(grid, CH_FAIRY_SRC, g.pending.fairy_src);

        // 스칼라
        let goal = g.goal_next();
        scal[0] = g.day as f32 / 100.0;
        scal[1] = (g.day % 10) as f32 / 10.0;
        scal[2] = if goal > 0 { (goal - g.day) as f32 / 10.0 } else { 0.0 };
        scal[3] = (g.day == goal) as u8 as f32;
        scal[4] = g.hearts as f32 / 36.0;
        scal[5] = g.hearts.min(10) as f32 / 10.0;
        scal[6] = (g.swaps.max(0) as f32).ln_1p() / 6.0;
        scal[7] = g.swaps.clamp(0, 10) as f32 / 10.0;
        scal[8] = g.achievements as f32 / 5.0;
        scal[9] = (g.day_off_day != 0) as u8 as f32;
        scal[10] = g.closed_day() as u8 as f32;
        scal[11] = if g.closed_day() { self.day_actions as f32 / DAY_OFF_CAP as f32 } else { 0.0 };
        scal[12] = g.deals_done as f32 / 10.0;
        scal[13] = g.deals_declined as f32 / 10.0;
        if let Some(i) = INPUT_PHASES.iter().position(|&p| p == g.phase) {
            scal[S_PHASE + i] = 1.0;
        }
        if let Some((offer, price)) = g.pending.devil {
            scal[S_DEVIL_PRICE] = price as f32 / 10.0;
            let (ty, kind, tier) = match offer {
                DevilOffer::Turret { col, tower } => {
                    mark(grid, CH_DEVIL, Some(tower));
                    grid[grid_index(CH_DEVIL, col, 0)] = 1.0;
                    (0, Some(Kind::ArrowTower), g.tiles[tower].tier)
                }
                DevilOffer::Slot { tile } => {
                    mark(grid, CH_DEVIL, Some(tile));
                    (1, None, 0)
                }
                DevilOffer::Iceberg { x, y, tier } => {
                    grid[grid_index(CH_DEVIL, x, y)] = 1.0;
                    (2, Some(Kind::Iceberg), tier)
                }
                DevilOffer::Place { kind, tier, tile } => {
                    mark(grid, CH_DEVIL, Some(tile));
                    (3, Some(kind), tier)
                }
            };
            scal[S_DEVIL_TYPE + ty] = 1.0;
            if let Some(k) = kind.and_then(kind_index) {
                scal[S_DEVIL_KIND + k] = 1.0;
            }
            scal[S_DEVIL_TIER] = tier as f32 / 4.0;
        }
        if let Some(item) = g.pending.shop {
            match item {
                ShopItem::Tile { kind, tier } => {
                    scal[S_SHOP_TYPE] = 1.0;
                    if let Some(k) = kind_index(kind) {
                        scal[S_SHOP_KIND + k] = 1.0;
                    }
                    scal[S_SHOP_TIER] = tier as f32 / 4.0;
                }
                ShopItem::Hearts(h) => {
                    scal[S_SHOP_TYPE + 1] = 1.0;
                    scal[S_SHOP_HEARTS] = h as f32 / 4.0;
                }
                ShopItem::TntBlast => scal[S_SHOP_TYPE + 2] = 1.0,
            }
        }
        if let Some((kind, tier)) = g.pending.place {
            if let Some(k) = kind_index(kind) {
                scal[S_PLACE_KIND + k] = 1.0;
            }
            scal[S_PLACE_TIER] = tier as f32 / 4.0;
        }
        debug_assert_eq!(S_PLACE_TIER + 1, N_SCALAR);
        debug_assert_eq!(N_KIND, 15);
    }
}
