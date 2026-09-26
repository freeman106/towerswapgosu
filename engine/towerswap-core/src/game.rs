//! 게임 상태와 낮 단계 로직. 원본 game.js(v120)의 함수 순서와 부작용(난수 소비, movedTime,
//! 상태 전환)을 그대로 따른다. 주석의 `이름()`은 원본의 난독화된 함수명이다.

use crate::kinds::{Kind, ALL_KINDS};
use crate::items::{DevilOffer, Pending};
use crate::night::NightState;
use crate::level::{first_level_terrain, Terrain, COLS, ROWS};
use crate::rng::JsRng;

pub type TileId = usize;

pub const MAX_HEARTS: i32 = 36; // ig
pub const START_HEARTS: i32 = 5;
const TILE: f64 = 48.0;

/// 원본 게임 상태(tK) 중 시뮬레이터가 쓰는 것
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,        // 0: 입력 대기
    Fall,        // 1: 낙하·합체 애니메이션
    Match,       // 2: 매치 표시 후 제거
    Dusk,        // 25: 스왑 소진, 밤 직전
    Night,       // 3
    DayOffOffer, // 108: 휴식일 제안
    Devil,       // 56: 악마 거래 수락/거절
    Merchant,    // 83: 팔 타일 선택(판매 강제)
    Shop,        // 79: 무료 품목 받기/거절
    PlaceItem,   // 43: 상점 품목 놓을 칸 선택
    Genie,       // 72: 자원 종류 선택
    TntMenu,     // 116: 전부 폭파 / 하나 폭파 / 닫기
    FairySource, // 109: 요정이 집을 타일 선택
    FairyTarget, // 109: 요정이 놓을 곳 선택
    GameOver,    // 11 (부활 없음)
}

impl Phase {
    pub fn code(self) -> i32 {
        match self {
            Phase::Idle => 0,
            Phase::Fall => 1,
            Phase::Match => 2,
            Phase::Dusk => 25,
            Phase::Night => 3,
            Phase::DayOffOffer => 108,
            Phase::Devil => 56,
            Phase::Merchant => 83,
            Phase::Shop => 79,
            Phase::PlaceItem => 43,
            Phase::Genie => 72,
            Phase::TntMenu => 116,
            Phase::FairySource | Phase::FairyTarget => 109,
            Phase::GameOver => 11,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tile {
    pub kind: Kind,
    pub tier: u8,
    pub gx: i32,
    pub gy: i32,
    pub moved_time: i64,
    pub dynamite: i32,
    pub flipped_preferred: bool,
    pub flipped: bool,
    pub frame: i32,
    pub removed: bool,
    pub match_join: Option<TileId>,
    pub upgraded: bool,
    pub turning_into: Option<Kind>,
    pub turning_into_tier: u8,
    pub dragged_off_edge: bool,
    pub alive: bool, // 원본 ak 목록에 있는지
    pub reload: f64, // 밤 재장전 (reloadDelay)
    pub born: (u64, i32), // 통계: 상자가 된 시점 (행동 수 stat_steps, 날)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    pub fn from_index(i: usize) -> Dir {
        [Dir::Up, Dir::Down, Dir::Left, Dir::Right][i]
    }
    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
        }
    }
}

/// 드래그 결과 종류 (`drag_plan`)
#[derive(Clone, Copy)]
enum DragPlan {
    Swap(TileId),
    Repair,
    Turret,
    Anvil(TileId),
    Toss,
    Move,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionResult {
    Ok,
    Invalid,
}

#[derive(Clone)]
pub struct Game {
    pub rng_v: JsRng, // 보드·상자
    pub rng_m: JsRng, // 적
    pub tiles: Vec<Tile>,
    pub ak: Vec<TileId>,                               // 원본 타일 목록(생성 순서)
    pub grid: [[Option<TileId>; ROWS + 1]; COLS + 1],  // ro[x][y]
    pub terrain: [[Terrain; ROWS + 1]; COLS + 1],       // eA
    pub turrets: [Option<TileId>; COLS + 1],            // iu[1..6]
    pub phase: Phase,
    pub day: i32,          // tr
    pub hearts: i32,       // il
    pub swaps: i64,        // r7
    pub achievements: i32, // sN
    pub boss_col: i32,     // rw
    pub day_off_day: i32,  // eU
    pub deals_done: i32,   // t_
    pub deals_declined: i32, // tg
    /// 통계: 등급별 상자 생성 수(합성·상점·악마 거래), 개봉 수. 게임 규칙에는 영향이 없다
    pub chest_made: [u32; 5],
    pub chest_opened: [u32; 5],
    /// 통계: 환경이 센 행동 수, 등급별 상자 보관 기간 합(생성→개봉, 행동 수와 날)
    pub stat_steps: u64,
    pub chest_hold: [(u64, u64); 5],
    pub combo: i32,        // rg
    pub loops: i64,        // ea.loops 대용 (순서만 의미 있음)
    // 스왑 직후 첫 매치 판정에서 우선하는 타일 (nV, nJ)
    swap_a: Option<TileId>,
    swap_b: Option<TileId>,
    // 매치 제거 직전의 타일 위치 (합체 애니메이션 시작점)
    start_pos: Vec<(i32, i32)>,
    pub night: NightState,
    pub(crate) ne_base: i64, // 원본 nE: 밤 사이에 이어지는 프레임 카운터
    pub trace_night: bool,   // 디버그: 밤 프레임마다 드래곤 상태를 stderr로
    pub(crate) night_start_loops: i64,
    pub pending: Pending,                  // 진행 중인 아이템 선택 정보
    pub(crate) opened_chest: Option<TileId>, // iy
    pub(crate) fairy_in_use: Option<TileId>, // ex (Fairy 타일만; 사용 후 제거 대기)
    pub(crate) fairy_check_did: bool,        // ex.matchCheckDid
    pub(crate) resume_phase: Phase,          // 선택 창을 닫을 때 돌아갈 단계
    pub(crate) fairy_swap_time: Option<i64>, // 요정 교환 시각(바로 다음 프레임의 요정 제거에 쓴다)
}

fn moveto(e: f64, a: f64, i: f64) -> f64 {
    let t = a - e;
    if t.abs() <= i {
        a
    } else {
        e + i * t.signum()
    }
}

impl Game {
    // ───────────────────────── 기본 조회 ─────────────────────────

    /// `_h`
    pub(crate) fn in_board(x: i32, y: i32) -> bool {
        x > 0 && y > 0 && x <= COLS as i32 && y <= ROWS as i32
    }

    /// `dk`
    pub fn terrain_at(&self, x: i32, y: i32) -> Terrain {
        if Self::in_board(x, y) {
            self.terrain[x as usize][y as usize]
        } else if x < 1 || x > COLS as i32 || y > 1 {
            Terrain::Water
        } else {
            Terrain::Grass
        }
    }

    /// `dT`
    pub(crate) fn is_slot(&self, x: i32, y: i32) -> bool {
        self.terrain_at(x, y) == Terrain::CannonSlot
    }

    /// `dC`
    pub(crate) fn is_ground(&self, x: i32, y: i32) -> bool {
        self.terrain_at(x, y) != Terrain::Water
    }

    /// `dw`
    pub fn tile_at(&self, x: i32, y: i32) -> Option<TileId> {
        if Self::in_board(x, y) {
            self.grid[x as usize][y as usize]
        } else {
            None
        }
    }

    pub(crate) fn set_grid(&mut self, x: i32, y: i32, v: Option<TileId>) {
        self.grid[x as usize][y as usize] = v;
    }

    /// `$b()`: 휴식일(또는 One Day 레벨의 닫힌 날)
    pub(crate) fn closed_day(&self) -> bool {
        self.day == self.day_off_day
    }

    fn upgrade_can(&self, id: TileId) -> bool {
        let t = &self.tiles[id];
        t.kind.upgraded().is_some() && t.tier < 4
    }

    // ───────────────────────── 타일 생성·삭제 ─────────────────────────

    /// `new ua` — 생성 즉시 목록 끝에 추가, movedTime = 현재 시각
    pub(crate) fn new_tile(&mut self, kind: Kind) -> TileId {
        let id = self.tiles.len();
        self.tiles.push(Tile {
            kind,
            tier: 0,
            gx: 0,
            gy: 0,
            moved_time: self.loops,
            dynamite: 0,
            flipped_preferred: false,
            flipped: false,
            frame: 1,
            removed: false,
            match_join: None,
            upgraded: false,
            turning_into: None,
            turning_into_tier: 0,
            dragged_off_edge: false,
            alive: true,
            reload: 0.0,
            born: (0, 0),
        });
        self.ak.push(id);
        id
    }

    /// `_delete2`
    pub(crate) fn delete_tile(&mut self, id: TileId) {
        self.tiles[id].match_join = None;
        if self.tiles[id].alive {
            self.tiles[id].alive = false;
            self.ak.retain(|&t| t != id);
        }
    }

    /// 목록(ak)·포탑·참조 어디에도 없는 타일을 지우고 번호를 다시 매긴다(남는 타일의 순서는 유지).
    /// `tiles`는 새 타일마다 커지므로 밤이 끝날 때마다 부른다. 밤 상태는 비운다.
    pub(crate) fn compact_tiles(&mut self) {
        const DEAD: usize = usize::MAX;
        self.night = Default::default();
        let mut map = vec![DEAD; self.tiles.len()];
        let mut mark = |t: Option<TileId>| {
            if let Some(t) = t {
                map[t] = 0;
            }
        };
        self.ak.iter().for_each(|&t| mark(Some(t)));
        self.turrets.iter().for_each(|&t| mark(t));
        self.tiles.iter().for_each(|t| mark(t.match_join));
        for t in [self.swap_a, self.swap_b, self.opened_chest, self.fairy_in_use] {
            mark(t);
        }
        let p = &self.pending;
        for t in [p.tnt_source, p.fairy, p.fairy_src] {
            mark(t);
        }
        if let Some((offer, _)) = p.devil {
            mark(match offer {
                DevilOffer::Turret { tower, .. } => Some(tower),
                DevilOffer::Slot { tile } | DevilOffer::Place { tile, .. } => Some(tile),
                DevilOffer::Iceberg { .. } => None,
            });
        }
        self.compact_with(map);
    }

    fn compact_with(&mut self, mut map: Vec<usize>) {
        let mut n = 0;
        for m in map.iter_mut() {
            if *m == 0 {
                *m = n;
                n += 1;
            }
        }
        let old = std::mem::take(&mut self.tiles);
        let old_start = std::mem::take(&mut self.start_pos);
        for (i, t) in old.into_iter().enumerate() {
            if map[i] != usize::MAX {
                self.tiles.push(t);
                if i < old_start.len() {
                    self.start_pos.push(old_start[i]); // 남는 타일 순서가 같으므로 앞부분이 된다
                }
            }
        }
        let re = |t: &mut Option<TileId>| {
            if let Some(v) = t {
                *v = map[*v];
            }
        };
        self.ak.iter_mut().for_each(|t| *t = map[*t]);
        self.grid.iter_mut().flatten().for_each(re);
        self.turrets.iter_mut().for_each(re);
        self.tiles.iter_mut().for_each(|t| re(&mut t.match_join));
        for t in [&mut self.swap_a, &mut self.swap_b, &mut self.opened_chest, &mut self.fairy_in_use] {
            re(t);
        }
        let p = &mut self.pending;
        for t in [&mut p.tnt_source, &mut p.fairy, &mut p.fairy_src] {
            re(t);
        }
        if let Some((offer, _)) = &mut p.devil {
            match offer {
                DevilOffer::Turret { tower: t, .. } | DevilOffer::Slot { tile: t } | DevilOffer::Place { tile: t, .. } => *t = map[*t],
                DevilOffer::Iceberg { .. } => {}
            }
        }
    }

    /// `kindSetRandomly`
    pub(crate) fn random_resource(&mut self) -> Kind {
        loop {
            let idx = self.rng_v.int(1, ALL_KINDS.len() as i64) as usize - 1;
            let k = ALL_KINDS[idx];
            if k.is_resource() && !k.unmovable() {
                return k;
            }
        }
    }

    /// `ua.prototype.upgrade`
    pub(crate) fn upgrade(&mut self, id: TileId) {
        let loops = self.loops;
        let t = &mut self.tiles[id];
        let up = t.kind.upgraded().expect("upgrade 불가 타일");
        t.upgraded = true;
        t.tier += 1;
        t.frame = 1;
        t.removed = false;
        if t.kind == Kind::Iron {
            t.flipped = false;
        }
        t.kind = up;
        t.match_join = None;
        if t.kind == Kind::Cannon && t.gx >= 5 {
            t.flipped = true;
            t.flipped_preferred = true;
        }
        if t.kind == Kind::FairyHouse {
            t.frame = 2;
        }
        t.moved_time = loops;
        if up == Kind::Chest {
            t.born = (self.stat_steps, self.day);
            let tier = t.tier.min(4) as usize;
            self.chest_made[tier] += 1;
        }
    }

    /// `removeFromGrid(e)`: 칸에서 빼고 위 타일을 한 칸씩 내린 뒤 맨 위에 새 타일 생성.
    /// `exclude_same`이 참이면 새 타일은 빠진 타일과 다른 종류로 뽑는다.
    pub(crate) fn remove_from_grid(&mut self, id: TileId, exclude_same: bool) {
        let t = self.tiles[id].gx;
        let mut s = self.tiles[id].gy;
        self.set_grid(t, s, None);
        if !self.is_slot(t, s) && self.is_ground(t, s) {
            self.phase = Phase::Fall; // _a(1)
            loop {
                s -= 1;
                if s < 1 {
                    break;
                }
                if self.is_slot(t, s) || !self.is_ground(t, s) {
                    return;
                }
                if let Some(i) = self.grid[t as usize][s as usize] {
                    self.set_grid(t, s, None);
                    self.tiles[i].gy += 1;
                    self.tiles[i].moved_time = self.loops;
                    self.set_grid(t, s + 1, Some(i));
                } else {
                    if s == 1 {
                        s -= 1;
                    }
                    break;
                }
            }
            if s == 0 {
                let exclude = if exclude_same { Some(self.tiles[id].kind) } else { None };
                self.spawn_top(t, exclude);
            }
        }
    }

    fn spawn_top(&mut self, x: i32, exclude: Option<Kind>) {
        if self.closed_day() {
            return;
        }
        let nid = self.new_tile(Kind::Wood);
        let mut k = self.random_resource();
        while Some(k) == exclude {
            k = self.random_resource();
        }
        let t = &mut self.tiles[nid];
        t.kind = k;
        t.gx = x;
        t.gy = 1;
        self.set_grid(x, 1, Some(nid));
        self.phase = Phase::Fall;
    }

    /// `removeDeleteAndReplace`
    pub(crate) fn remove_delete_and_replace(&mut self, id: TileId, exclude_same: bool) {
        self.remove_from_grid(id, exclude_same);
        self.delete_tile(id);
    }

    /// 통계 카운터를 비운다 (저장한 상태에서 새 에피소드를 시작할 때)
    pub fn reset_stats(&mut self) {
        self.chest_made = [0; 5];
        self.chest_opened = [0; 5];
        self.chest_hold = [(0, 0); 5];
        self.stat_steps = 0;
        let day = self.day;
        for t in self.tiles.iter_mut() {
            t.born = (0, day);
        }
    }

    // ───────────────────────── 새 게임 ─────────────────────────

    /// 새 게임: `gz()`(보드 생성) → `_X(!0)` → 자동 매치 해소까지 진행
    pub fn new_game(vseed: u32, mseed: u32) -> Game {
        let mut g = Game {
            rng_v: JsRng::new(vseed),
            rng_m: JsRng::new(mseed),
            tiles: Vec::new(),
            ak: Vec::new(),
            grid: [[None; ROWS + 1]; COLS + 1],
            terrain: first_level_terrain(),
            turrets: [None; COLS + 1],
            phase: Phase::Match,
            day: 1,
            hearts: START_HEARTS,
            swaps: 5,
            achievements: 0,
            boss_col: 0,
            day_off_day: 0,
            deals_done: 0,
            deals_declined: 0,
            chest_made: [0; 5],
            chest_opened: [0; 5],
            stat_steps: 0,
            chest_hold: [(0, 0); 5],
            combo: 0,
            loops: 0,
            swap_a: None,
            swap_b: None,
            start_pos: Vec::new(),
            night: NightState::default(),
            ne_base: 10_000,
            trace_night: std::env::var("TS_TRACE").is_ok(),
            night_start_loops: 0,
            pending: Pending::default(),
            opened_chest: None,
            fairy_in_use: None,
            fairy_check_did: false,
            resume_phase: Phase::Idle,
            fairy_swap_time: None,
        };
        // gz: 행 우선으로 땅 칸을 채운다
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if !g.is_ground(x, y) || g.is_slot(x, y) {
                    continue;
                }
                let id = g.new_tile(Kind::Wood);
                let k = g.random_resource();
                let t = &mut g.tiles[id];
                t.kind = k;
                t.gx = x;
                t.gy = y;
                g.set_grid(x, y, Some(id));
            }
        }
        // _X(!0) → 상태 2
        g.phase = Phase::Match;
        g.advance();
        g
    }

    /// `lR(board)`: 원본 보드 문자열을 불러온다(시나리오 검사용).
    /// 형식: 포탑 6칸 + 1..7행 × 6칸, 칸마다 (id, tier) 2글자. 'o'=물, 'e'=빈 땅,
    /// 'C'/'D'=cannon slot(+tier>0이면 대포, D는 왼쪽), 'f'=왼쪽 대포, 'I'=빙산(물 칸).
    /// 불러온 뒤 원본 시나리오와 같이 스왑 5, 콤보 0, 입력 대기 상태로 둔다.
    pub fn load_board(&mut self, s: &str) {
        let b: Vec<char> = s.chars().collect();
        assert_eq!(b.len(), 12 + 2 * COLS * ROWS, "보드 문자열 길이");
        let kind_of = |c: char| ALL_KINDS.iter().copied().find(|k| k.id() == c);
        self.tiles.clear();
        self.ak.clear();
        self.turrets = [None; COLS + 1];
        let mut t = 0;
        for x in 1..=COLS {
            let (c, l) = (b[t], b[t + 1].to_digit(10).unwrap_or(0) as u8);
            t += 2;
            if c != 'e' {
                let id = self.new_tile(kind_of(c).expect("포탑 종류"));
                self.delete_tile(id); // 포탑은 타일 목록에 없다
                let tile = &mut self.tiles[id];
                tile.tier = l;
                tile.gx = x as i32;
                tile.gy = 0;
                self.turrets[x] = Some(id);
            }
        }
        for y in 1..=ROWS {
            for x in 1..=COLS {
                self.grid[x][y] = None;
                let (c, l) = (b[t], b[t + 1].to_digit(10).unwrap_or(0) as u8);
                t += 2;
                match c {
                    'o' => {
                        self.terrain[x][y] = Terrain::Water;
                        continue;
                    }
                    'e' => {
                        self.terrain[x][y] = Terrain::Grass;
                        continue;
                    }
                    _ => {}
                }
                let (kind, slot, left) = match c {
                    'f' => (Kind::Cannon, false, true),
                    'D' => (Kind::Cannon, true, true),
                    'C' => (Kind::Cannon, true, false),
                    _ => (kind_of(c).expect("타일 종류"), false, false),
                };
                self.terrain[x][y] = if kind == Kind::Iceberg { Terrain::Water } else { Terrain::Grass };
                if slot {
                    self.terrain[x][y] = Terrain::CannonSlot;
                    if l == 0 {
                        continue;
                    }
                }
                let id = self.new_tile(kind);
                self.grid[x][y] = Some(id);
                let tile = &mut self.tiles[id];
                tile.gx = x as i32;
                tile.gy = y as i32;
                tile.tier = l;
                if kind == Kind::Chest && l == 0 {
                    tile.tier = 1;
                    tile.removed = true;
                }
                if kind == Kind::Tnt {
                    tile.dynamite = l as i32;
                    tile.tier = 0;
                }
                if kind == Kind::FairyHouse {
                    tile.frame = (l as i32).min(2);
                    tile.tier = 1;
                }
                if kind == Kind::Anvil && tile.tier < 1 {
                    tile.tier = 1;
                }
                if left {
                    tile.flipped = true;
                    tile.flipped_preferred = true;
                }
            }
        }
        self.phase = Phase::Idle;
        self.swaps = 5;
        self.combo = 0;
    }

    // ───────────────────────── 매치 판정·해소 ─────────────────────────

    /// `dE(e)`: e에서 오른쪽·아래로 같은 (종류, tier) 줄을 찾아 표시한다.
    fn detect_from(&mut self, e: TileId) {
        if !self.upgrade_can(e) {
            return;
        }
        let (ex, ey) = (self.tiles[e].gx, self.tiles[e].gy);
        if self.is_slot(ex, ey) {
            return;
        }
        let (ekind, etier) = (self.tiles[e].kind, self.tiles[e].tier);
        for l in 1..=2 {
            let (mut s, mut r) = (ex, ey);
            let mut run: Vec<TileId> = Vec::new();
            let mut i = Some(e);
            loop {
                run.push(i.unwrap());
                if l == 1 {
                    s += 1
                } else {
                    r += 1
                }
                i = self.tile_at(s, r);
                match i {
                    Some(ii) => {
                        let ti = &self.tiles[ii];
                        if ti.kind.unmovable() || Some(ii) == self.fairy_in_use || self.is_slot(ti.gx, ti.gy) || ti.kind != ekind || ti.tier != etier {
                            break;
                        }
                    }
                    None => break,
                }
            }
            if run.len() >= 3 {
                self.phase = Phase::Match; // tK = 2
                let mut n = e;
                for &t in &run {
                    if self.tiles[t].moved_time > self.tiles[n].moved_time {
                        n = t;
                    }
                }
                for &t in &run {
                    if Some(t) == self.swap_a || Some(t) == self.swap_b {
                        n = t;
                    }
                }
                for &t in &run {
                    if let Some(j) = self.tiles[t].match_join {
                        n = j;
                        break;
                    }
                }
                for &t in &run {
                    if let Some(j) = self.tiles[t].match_join {
                        if j != n {
                            // 다른 그룹과 합치기: 목록 전체에서 같은 join을 가진 타일을 n으로
                            for &o in &self.ak.clone() {
                                if o != t && self.tiles[o].match_join == Some(j) {
                                    self.tiles[o].match_join = Some(n);
                                }
                            }
                        }
                    }
                    self.tiles[t].match_join = Some(n);
                    self.tiles[t].removed = true;
                }
            }
        }
    }

    /// `dy(e)`: e를 join으로 가진 보드 위 타일 수
    fn group_size(&self, e: TileId) -> i64 {
        let mut n = 0;
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if let Some(t) = self.tile_at(x, y) {
                    if self.tiles[t].match_join == Some(e) {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// `merchantAndMatch4SwapsGet`
    pub fn tile_value(&self, id: TileId) -> i64 {
        let t = &self.tiles[id];
        match t.kind {
            Kind::Chest => match t.tier {
                1 => 4,
                2 => 16,
                3 => 90,
                4 => 420,
                _ => 3i64.pow(t.tier as u32),
            },
            Kind::Fairy => 9,
            Kind::Tnt => 3 * t.dynamite as i64,
            Kind::FairyHouse => 27,
            _ => 3i64.pow(t.tier as u32),
        }
    }

    /// `gQ`: 스왑 추가
    pub(crate) fn add_swaps(&mut self, n: i64) {
        self.swaps += n;
    }

    /// `dG()`: 매치를 찾아 표시한다. 찾으면 참. 못 찾으면 `m$()`로 이어진다.
    pub(crate) fn detect_matches(&mut self) -> bool {
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if let Some(a) = self.tile_at(x, y) {
                    if Some(a) != self.fairy_in_use {
                        self.detect_from(a);
                    }
                }
            }
        }
        if self.phase == Phase::Match {
            for y in 1..=ROWS as i32 {
                for x in 1..=COLS as i32 {
                    if let Some(a) = self.tile_at(x, y) {
                        if self.tiles[a].match_join == Some(a) {
                            // d3: 큰 매치 보너스
                            let n = self.group_size(a);
                            if n > 3 && !self.closed_day() {
                                let v = (n - 3) * self.tile_value(a);
                                self.add_swaps(v);
                            }
                            self.combo += 1;
                        }
                    }
                }
            }
        }
        if self.phase == Phase::Idle {
            self.after_resolution();
        }
        self.phase == Phase::Match
    }

    /// `m$()`: 해소가 끝났을 때. 콤보 보너스, 스왑 소진 시 밤 진입.
    pub(crate) fn after_resolution(&mut self) {
        if self.phase != Phase::Idle {
            return;
        }
        if !self.closed_day() && self.combo >= 3 {
            let a = (self.combo - 3 + 1) as i64;
            self.add_swaps(a);
        }
        self.combo = 0;
        if self.swaps <= 0 {
            self.phase = Phase::Dusk;
        }
    }

    /// 상태 2 종료: 매치된 타일 제거(행 우선), 중력·생성
    fn match_phase_end(&mut self) {
        // 제거 전 위치 기록 (합체 애니메이션 시작점)
        let start: Vec<(i32, i32)> = self.tiles.iter().map(|t| (t.gx, t.gy)).collect();
        self.start_pos = start;
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if let Some(a) = self.tile_at(x, y) {
                    if !self.tiles[a].removed {
                        continue;
                    }
                    if let Some(k) = self.tiles[a].turning_into {
                        let tier = self.tiles[a].turning_into_tier;
                        let t = &mut self.tiles[a];
                        t.kind = k;
                        t.frame = 1;
                        t.removed = false;
                        t.tier = tier;
                        t.turning_into = None;
                    } else {
                        if self.tiles[a].match_join != Some(a) {
                            self.remove_from_grid(a, false);
                        }
                        if self.tiles[a].match_join.is_none() {
                            self.delete_tile(a);
                        }
                        self.phase = Phase::Fall;
                    }
                }
            }
        }
        self.phase = Phase::Fall;
    }

    /// 상태 1: 합체 애니메이션을 원본 이동식대로 돌려 업그레이드 시각을 정하고, 끝나면 정리한다.
    fn fall_phase_run(&mut self) {
        let f0 = self.loops;
        // join 별 그룹 모으기
        let joins: Vec<TileId> = self
            .ak
            .iter()
            .copied()
            .filter(|&t| self.tiles[t].match_join == Some(t))
            .collect();
        for j in joins {
            let members: Vec<TileId> = self
                .ak
                .iter()
                .copied()
                .filter(|&t| t == j || (self.tiles[t].removed && self.tiles[t].match_join == Some(j)))
                .collect();
            let pos = |g: &Game, id: TileId| -> (f64, f64) {
                let (x, y) = g.start_pos.get(id).copied().unwrap_or((g.tiles[id].gx, g.tiles[id].gy));
                ((x - 1) as f64 * TILE, (y - 1) as f64 * TILE)
            };
            let (jx, mut jy) = pos(self, j);
            let jty = (self.tiles[j].gy - 1) as f64 * TILE;
            let mut mp: Vec<(TileId, f64, f64, bool)> =
                members.iter().filter(|&&t| t != j).map(|&t| {
                    let (x, y) = pos(self, t);
                    (t, x, y, false)
                }).collect();
            let mut arrived_frame: Option<i64> = None;
            let mut f = 0i64;
            while arrived_frame.is_none() {
                f += 1;
                assert!(f < 10_000, "합체 애니메이션이 끝나지 않음");
                let mut mi = 0;
                for &t in &members {
                    if t == j {
                        if jy < jty {
                            jy = moveto(jy, jty, 2.88);
                        }
                        continue;
                    }
                    let m = &mut mp[mi];
                    mi += 1;
                    if m.3 {
                        continue;
                    }
                    m.1 = moveto(m.1, jx, 0.1 * (m.1 - jx).abs() + 0.96);
                    let mut c = 0.09;
                    if m.2 > jy {
                        c /= 2.0;
                    }
                    m.2 = moveto(m.2, jy, (m.2 - jy).abs() * c + 2.88);
                    if !((m.1 - jx).abs() > 1.44 || (m.2 - jy).abs() > 1.44) {
                        m.3 = true;
                        if arrived_frame.is_none() {
                            arrived_frame = Some(f);
                        }
                    }
                }
            }
            self.loops = f0 + arrived_frame.unwrap();
            self.upgrade(j);
            for (t, ..) in mp {
                self.delete_tile(t);
            }
        }
        // 상태 1 종료 처리
        for id in self.ak.clone() {
            if self.tiles[id].match_join == Some(id) {
                self.upgrade(id);
            }
            if self.tiles[id].removed {
                self.delete_tile(id);
            }
            self.tiles[id].upgraded = false;
        }
        self.loops = f0 + 1000;
    }

    /// 입력을 기다리는 상태(또는 밤 직전)가 될 때까지 해소를 진행한다.
    pub fn advance(&mut self) {
        loop {
            match self.phase {
                Phase::Match => {
                    self.loops += 100;
                    self.match_phase_end();
                }
                Phase::Fall => {
                    self.fall_phase_run();
                    if let Some(ex) = self.fairy_in_use {
                        if self.fairy_check_did {
                            // 요정 사용 후 매치까지 끝났으면 요정 타일 제거 (중력·생성 → 다시 상태 1)
                            self.fairy_in_use = None;
                            self.fairy_check_did = false;
                            self.remove_delete_and_replace(ex, false);
                            if self.phase == Phase::Fall {
                                continue;
                            }
                            self.phase = Phase::Idle;
                            self.detect_matches();
                            continue;
                        }
                        self.fairy_check_did = true;
                        self.phase = Phase::Idle;
                        if !self.detect_matches() {
                            if let Some(ex) = self.fairy_in_use.take() {
                                self.fairy_check_did = false;
                                let saved = self.loops;
                                if let Some(t) = self.fairy_swap_time.take() {
                                    self.loops = t;
                                }
                                self.remove_delete_and_replace(ex, false);
                                self.loops = saved;
                                if self.phase != Phase::Fall {
                                    self.detect_matches();
                                }
                            }
                        }
                        continue;
                    }
                    self.fairy_swap_time = None;
                    self.phase = Phase::Idle;
                    self.detect_matches();
                }
                _ => return,
            }
        }
    }

    // ───────────────────────── 낮 행동: 드래그 ─────────────────────────

    /// 드래그 판정: 유효하면 (끄는 타일, 결과). 상태를 바꾸지 않는다(행동 마스크용).
    fn drag_plan(&self, x: i32, y: i32, dir: Dir) -> Option<(TileId, DragPlan)> {
        if self.phase != Phase::Idle {
            return None;
        }
        let nq = self.tile_at(x, y)?;
        if self.tiles[nq].kind.unmovable() {
            return None;
        }
        let (nu, nb) = dir.delta();
        // $C
        let anvil = self.anvil_target(nq, nu, nb, true);
        let mut nz = None;
        if anvil.is_none() {
            nz = self.tile_at(x + nu, y + nb);
            if let Some(z) = nz {
                if (self.tiles[z].kind == Kind::Iceberg) != (self.tiles[nq].kind == Kind::Iceberg) {
                    nz = None;
                }
            }
        }
        // d9 내부 검사
        if nz.is_none() {
            if let Some(s) = self.tile_at(x + nu, y + nb) {
                if self.tiles[s].kind == Kind::Anvil && anvil.is_none() {
                    return None;
                }
            }
        }
        // dR
        if let Some(z) = nz {
            return (z != nq && self.swap_valid(nq, z)).then_some((nq, DragPlan::Swap(z)));
        }
        let (ex, ey) = (x + nu, y + nb);
        let kind = self.tiles[nq].kind;
        let plan = if kind == Kind::Stone && self.hearts < MAX_HEARTS && !self.closed_day() && y == 1 && nb < 0 {
            DragPlan::Repair // 성 보수
        } else if nb < 0 && y == 1 && self.turret_can_arm(nq, x) {
            DragPlan::Turret // 성 포탑 (mW)
        } else if let Some(t) = anvil {
            DragPlan::Anvil(t) // 모루 (dX → 확인창 → Upgrade)
        } else if if kind == Kind::Iceberg { !Self::in_board(ex, ey) } else { !self.is_ground(ex, ey) } {
            DragPlan::Toss // 버리기 (dj → dz)
        } else if self.is_ground(ex, ey) == (kind != Kind::Iceberg) && Self::in_board(ex, ey) && self.tile_at(ex, ey).is_none() {
            if self.is_slot(ex, ey) && kind != Kind::Cannon {
                return None;
            }
            DragPlan::Move // 빈 칸으로 이동 (dU)
        } else {
            return None;
        };
        Some((nq, plan))
    }

    /// 드래그가 유효한지 (상태 변화 없음)
    pub fn drag_valid(&self, x: i32, y: i32, dir: Dir) -> bool {
        self.drag_plan(x, y, dir).is_some()
    }

    /// 드래그(스왑·이동·버리기·성 보수·포탑·모루). 무효면 아무 일도 없다.
    pub fn drag(&mut self, x: i32, y: i32, dir: Dir) -> ActionResult {
        if self.phase != Phase::Idle {
            return ActionResult::Invalid;
        }
        let Some(nq) = self.tile_at(x, y) else { return ActionResult::Invalid };
        if self.tiles[nq].kind.unmovable() {
            return ActionResult::Invalid;
        }
        self.loops += 1000;
        let Some((_, plan)) = self.drag_plan(x, y, dir) else { return ActionResult::Invalid };
        let (nu, nb) = dir.delta();
        let (ex, ey) = (x + nu, y + nb);
        match plan {
            DragPlan::Swap(z) => {
                // dB
                let (ax, ay, bx, by) = (self.tiles[nq].gx, self.tiles[nq].gy, self.tiles[z].gx, self.tiles[z].gy);
                self.set_grid(bx, by, Some(nq));
                self.set_grid(ax, ay, Some(z));
                self.tiles[nq].gx = bx;
                self.tiles[nq].gy = by;
                self.tiles[z].gx = ax;
                self.tiles[z].gy = ay;
                self.tiles[nq].moved_time = self.loops + 1;
                self.tiles[z].moved_time = self.loops;
                self.spend_swap(Some(nq), Some(z));
            }
            DragPlan::Repair => {
                self.remove_delete_and_replace(nq, true);
                self.spend_swap(None, None);
                self.hearts = (self.hearts + 1).min(MAX_HEARTS);
            }
            DragPlan::Turret => {
                self.remove_delete_and_replace(nq, false);
                self.tiles[nq].alive = false;
                self.tiles[nq].gx = x;
                self.tiles[nq].gy = 0;
                self.turrets[x as usize] = Some(nq);
                self.spend_swap(None, None);
            }
            DragPlan::Anvil(t) => {
                let (tx, ty) = (self.tiles[t].gx, self.tiles[t].gy);
                if self.can_go_here(self.tiles[nq].kind, tx, ty) {
                    self.remove_from_grid(nq, false);
                    // 모루가 A 바로 위에 있었다면 방금 중력으로 한 칸 내려왔다. 원본은 내려온 위치를 쓴다.
                    let (tx, ty) = (self.tiles[t].gx, self.tiles[t].gy);
                    self.tiles[nq].gx = tx;
                    self.tiles[nq].gy = ty;
                    self.tiles[nq].moved_time = self.loops;
                    self.set_grid(tx, ty, Some(nq));
                } else {
                    self.remove_from_grid(t, false);
                }
                self.delete_tile(t);
                self.upgrade(nq);
                self.phase = Phase::Fall;
                self.spend_swap(Some(nq), None);
            }
            DragPlan::Toss => {
                self.remove_from_grid(nq, false);
                self.tiles[nq].removed = true;
                self.tiles[nq].dragged_off_edge = true;
                self.spend_swap(None, None);
            }
            DragPlan::Move => {
                self.move_into_empty(nq, ex, ey);
                self.spend_swap(Some(nq), None);
            }
        }
        ActionResult::Ok
    }

    /// `gq`: 스왑 유효성
    pub(crate) fn swap_valid(&self, a: TileId, b: TileId) -> bool {
        let (ta, tb) = (&self.tiles[a], &self.tiles[b]);
        (ta.kind == Kind::Iceberg) == (tb.kind == Kind::Iceberg)
            && (ta.kind != tb.kind || ta.tier != tb.tier)
            && (!(self.is_slot(ta.gx, ta.gy) || self.is_slot(tb.gx, tb.gy))
                || (ta.kind == Kind::Cannon && tb.kind == Kind::Cannon))
    }

    /// `dX`: nq를 모루로 끌어 업그레이드할 수 있으면 그 모루
    fn anvil_target(&self, nq: TileId, nu: i32, nb: i32, _in_c: bool) -> Option<TileId> {
        if !self.upgrade_can(nq) || self.tiles[nq].kind == Kind::Anvil {
            return None;
        }
        let a = self.tile_at(self.tiles[nq].gx + nu, self.tiles[nq].gy + nb)?;
        if self.tiles[a].kind == Kind::Anvil && self.tiles[nq].tier as i32 == self.tiles[a].tier as i32 - 1 {
            Some(a)
        } else {
            None
        }
    }

    /// `armCastleTowerCan` + `mH`
    pub(crate) fn turret_can_arm(&self, nq: TileId, x: i32) -> bool {
        let t = &self.tiles[nq];
        if t.kind != Kind::ArrowTower || !(x == 1 || x == 6) {
            return false;
        }
        match self.turrets[x as usize] {
            None => true,
            Some(o) => (self.tiles[o].tier) < t.tier,
        }
    }

    /// `us.prototype.canGoHere` (행 > 0)
    pub(crate) fn can_go_here(&self, kind: Kind, x: i32, y: i32) -> bool {
        if !Self::in_board(x, y) {
            return false;
        }
        if kind == Kind::Iceberg {
            !self.is_ground(x, y)
        } else {
            self.is_ground(x, y) && (!self.is_slot(x, y) || kind == Kind::Cannon)
        }
    }

    /// `dU`: 빈 칸으로 이동 후 아래로 낙하
    pub(crate) fn move_into_empty(&mut self, nq: TileId, e: i32, mut a: i32) {
        self.remove_from_grid(nq, false);
        self.tiles[nq].gx = e;
        self.tiles[nq].gy = a;
        self.tiles[nq].moved_time = self.loops;
        self.set_grid(e, a, Some(nq));
        if self.is_slot(e, a) {
            return;
        }
        loop {
            a += 1;
            if a > ROWS as i32 || !self.is_ground(e, a) || self.tile_at(e, a).is_some() {
                a -= 1;
                break;
            }
        }
        if a > self.tiles[nq].gy {
            self.remove_from_grid(nq, false);
            self.tiles[nq].gy = a;
            self.set_grid(e, a, Some(nq));
        }
    }

    /// `dV`: 스왑 1 소모 후 매치 판정
    fn spend_swap(&mut self, a: Option<TileId>, b: Option<TileId>) {
        self.swap_a = a;
        self.swap_b = b;
        if !self.closed_day() {
            self.swaps -= 1;
        }
        if self.phase != Phase::Fall {
            self.detect_matches();
        }
        self.after_resolution();
        self.swap_a = None;
        self.swap_b = None;
        self.advance();
    }

    // ───────────────────────── 출력 ─────────────────────────

    /// 원본 대조용 보드 문자열 (oracle/driver.js의 state()와 같은 형식)
    pub fn board_cells(&self) -> Vec<Vec<String>> {
        let mut rows = Vec::new();
        for y in 1..=ROWS as i32 {
            let mut row = Vec::new();
            for x in 1..=COLS as i32 {
                match self.tile_at(x, y) {
                    None => row.push(if self.is_ground(x, y) { "..".to_string() } else { "~~".to_string() }),
                    Some(t) => {
                        let t = &self.tiles[t];
                        let mut s = format!("{}{}", t.kind.id(), if t.kind == Kind::Tnt { t.dynamite } else { t.tier as i32 });
                        if t.kind == Kind::Cannon {
                            s.push(if t.flipped_preferred { '<' } else { '>' });
                        }
                        row.push(s);
                    }
                }
            }
            rows.push(row);
        }
        rows
    }

    pub fn turret_cells(&self) -> Vec<String> {
        (1..=COLS)
            .map(|x| match self.turrets[x] {
                Some(t) => format!("{}{}", self.tiles[t].kind.id(), self.tiles[t].tier),
                None => String::new(),
            })
            .collect()
    }
}
