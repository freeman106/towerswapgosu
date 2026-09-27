//! 상자 열기와 아이템(악마 거래, TNT, 모루, 상인, 상점, 지니, 요정), 탭 행동.
//! 원본 treasureChestOpen과 상태 56/57/72/79/83/43/116/58/109의 흐름과 난수 소비 순서를 따른다.

use crate::game::{ActionResult, Game, Phase, TileId, MAX_HEARTS};
use crate::kinds::{Kind, ALL_KINDS};
use crate::level::{Terrain, COLS, ROWS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevilOffer {
    /// 보드의 화살탑을 빈 포탑 슬롯으로
    Turret { col: i32, tower: TileId },
    /// 그 칸을 대포 슬롯으로 (대포가 아니면 타일 제거)
    Slot { tile: TileId },
    /// 빈 물 칸에 빙산
    Iceberg { x: i32, y: i32, tier: u8 },
    /// 타일을 제안 종류·등급으로 바꿈
    Place { kind: Kind, tier: u8, tile: TileId },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShopItem {
    Tile { kind: Kind, tier: u8 },
    Hearts(i32),
    TntBlast,
}

#[derive(Clone, Debug, Default)]
pub struct Pending {
    pub devil: Option<(DevilOffer, i32)>, // (제안, 하트 가격)
    pub shop: Option<ShopItem>,            // 무료 품목
    pub place: Option<(Kind, u8)>,         // 놓을 품목
    pub tnt_source: Option<TileId>,        // TNT 타일 (None: 상점 폭파)
    pub tnt_cancelable: bool,
    pub fairy: Option<TileId>,     // 사용 중인 Fairy / Fairy House
    pub fairy_src: Option<TileId>, // 요정이 집은 타일
}

/// `mZ`: 상자 등급별 스왑
pub fn chest_swaps(tier: u8) -> i64 {
    match tier {
        1 => 2,
        2 => 12,
        3 => 70,
        4 => 380,
        _ => 0,
    }
}

/// `mq`: 아이템 확률 (원본처럼 2.9로 연달아 나눈다)
fn item_chance(tier: u8) -> f64 {
    let mut a = 1.0;
    if tier <= 3 {
        a /= 2.9;
    }
    if tier <= 2 {
        a /= 2.9;
    }
    if tier <= 1 {
        a /= 2.9;
    }
    a
}

impl Game {
    // ───────────────────────── 유효성 (행동 마스크용) ─────────────────────────

    pub fn merchant_can_sell(&self, x: i32, y: i32) -> bool {
        self.phase == Phase::Merchant && self.tile_at(x, y).map_or(false, |t| Some(t) != self.opened_chest)
    }

    pub fn can_place_public(&self, kind: Kind, tier: u8, x: i32, y: i32) -> bool {
        (1..=COLS as i32).contains(&x) && (0..=ROWS as i32).contains(&y) && self.can_place(kind, tier, x, y)
    }

    pub fn tnt_can_target(&self, x: i32, y: i32) -> bool {
        self.phase == Phase::TntMenu
            && self.tile_at(x, y).map_or(false, |t| self.tiles[t].tier < 1 && Some(t) != self.pending.tnt_source)
    }

    pub fn fairy_can_source(&self, x: i32, y: i32) -> bool {
        self.phase == Phase::FairySource && self.tile_at(x, y).map_or(false, |t| self.tiles[t].kind != Kind::Fairy)
    }

    pub fn fairy_can_target(&self, x: i32, y: i32) -> bool {
        self.phase == Phase::FairyTarget && self.fairy_target_ok(self.pending.fairy_src.unwrap(), x, y)
    }

    /// 요정이 src 타일을 (x, y)에 놓을 수 있는지 (단계 검사 없음)
    pub(crate) fn fairy_target_ok(&self, src: TileId, x: i32, y: i32) -> bool {
        if !(1..=COLS as i32).contains(&x) || !(0..=ROWS as i32).contains(&y) {
            return false;
        }
        match if y >= 1 { self.tile_at(x, y) } else { None } {
            Some(z) => self.swap_valid(src, z),
            None => self.can_place(self.tiles[src].kind, self.tiles[src].tier, x, y),
        }
    }

    /// 요정이 src를 놓을 곳이 하나라도 있는지
    pub fn fairy_source_has_target(&self, src: TileId) -> bool {
        (0..=ROWS as i32).any(|y| (1..=COLS as i32).any(|x| self.fairy_target_ok(src, x, y)))
    }

    /// 요정으로 할 수 있는 이동이 하나라도 있는지 (집을 타일 × 놓을 곳)
    pub fn fairy_any_move(&self) -> bool {
        (1..=ROWS as i32).any(|y| {
            (1..=COLS as i32).any(|x| {
                self.tile_at(x, y).map_or(false, |t| self.tiles[t].kind != Kind::Fairy && self.fairy_source_has_target(t))
            })
        })
    }

    /// 탭이 유효한지 (상태 변화 없음)
    pub fn tap_valid(&self, x: i32, y: i32) -> bool {
        if !matches!(self.phase, Phase::Idle | Phase::Dusk) {
            return false;
        }
        let Some(t) = self.tile_at(x, y) else { return false };
        match self.tiles[t].kind {
            Kind::Chest => !self.closed_day(),
            Kind::Cannon | Kind::Tnt | Kind::Fairy => true,
            Kind::FairyHouse => self.tiles[t].frame > 1,
            _ => false,
        }
    }

    // ───────────────────────── 탭 ─────────────────────────

    /// 무료 탭 행동: 상자 열기, 대포 방향 전환, TNT 메뉴, 요정 사용
    pub fn tap(&mut self, x: i32, y: i32) -> ActionResult {
        if !self.tap_valid(x, y) {
            return ActionResult::Invalid;
        }
        let t = self.tile_at(x, y).unwrap();
        match self.tiles[t].kind {
            Kind::Chest => self.open_chest(t),
            Kind::Cannon => {
                let tile = &mut self.tiles[t];
                tile.flipped_preferred = !tile.flipped_preferred;
                tile.flipped = tile.flipped_preferred;
            }
            Kind::Tnt => {
                self.resume_phase = self.phase;
                self.pending.tnt_source = Some(t);
                self.pending.tnt_cancelable = true;
                self.phase = Phase::TntMenu;
            }
            _ => self.fairy_start(t), // Fairy, 충전된 FairyHouse
        }
        ActionResult::Ok
    }

    // ───────────────────────── 상자 ─────────────────────────

    /// `treasureChestOpen`
    fn open_chest(&mut self, t: TileId) {
        let tier = self.tiles[t].tier;
        let (bs, bd) = self.tiles[t].born;
        let h = &mut self.chest_hold[tier.min(4) as usize];
        h.0 += self.stat_steps - bs.min(self.stat_steps);
        h.1 += (self.day - bd).max(0) as u64;
        self.chest_opened[tier.min(4) as usize] += 1;
        if tier > 0 {
            self.add_swaps(chest_swaps(tier));
        }
        self.tiles[t].removed = true;
        self.phase = Phase::Match;
        if self.rng_v.float() < item_chance(tier) {
            let p = 0.55 * 0.9f64.powf(self.deals_done as f64);
            let a = if self.rng_v.float() < p { 1 } else { self.rng_v.int(2, 7) };
            self.opened_chest = Some(t);
            self.tiles[t].turning_into_tier = 0;
            match a {
                1 => self.devil_prepare(t),
                2 => {
                    self.tiles[t].turning_into = Some(Kind::Tnt);
                    self.tiles[t].dynamite = self.rng_v.int(1, 3) as i32;
                }
                3 => {
                    self.tiles[t].turning_into = Some(Kind::Anvil);
                    self.tiles[t].turning_into_tier = 1;
                    if tier > 2 {
                        self.tiles[t].turning_into_tier = self.rng_v.int(1, tier as i64 - 1) as u8;
                    }
                }
                4 => self.phase = Phase::Merchant,
                5 => {
                    self.shop_prepare();
                    self.phase = Phase::Shop;
                }
                6 => self.phase = Phase::Genie,
                _ => self.tiles[t].turning_into = Some(Kind::Fairy),
            }
        }
        if self.phase == Phase::Match {
            self.opened_chest = None;
            self.advance();
        }
    }

    /// `dA(e)`: 최종 종류 {Anvil, Ballista, ArrowTower, Cannon, Chest, IceWall} 중 무작위(제외 가능)
    fn random_final_kind(&mut self, exclude: Option<Kind>) -> Kind {
        loop {
            let k = ALL_KINDS[self.rng_v.int(1, ALL_KINDS.len() as i64) as usize - 1];
            if k.upgraded() == Some(k) && !k.unmovable() && k != Kind::Iceberg && Some(k) != exclude {
                return k;
            }
        }
    }

    /// `$T`: 빙산을 놓을 수 있는 빈 물 칸 (열 우선 순서)
    fn iceberg_cells(&self) -> Vec<(i32, i32)> {
        let mut v = Vec::new();
        for x in 1..=COLS as i32 {
            for y in 1..=ROWS as i32 {
                if self.can_go_here(Kind::Iceberg, x, y) && self.tile_at(x, y).is_none() {
                    v.push((x, y));
                }
            }
        }
        v
    }

    fn pick_iceberg_cell(&mut self) -> Option<(i32, i32)> {
        let cells = self.iceberg_cells();
        if cells.is_empty() {
            None
        } else {
            Some(cells[self.rng_v.item_index(cells.len())])
        }
    }

    /// `dF`: 상하좌우 이웃 중 kind인 타일 수
    fn neighbor_count(&self, x: i32, y: i32, kind: Kind) -> i32 {
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .filter(|(dx, dy)| self.tile_at(x + dx, y + dy).map_or(false, |t| self.tiles[t].kind == kind))
            .count() as i32
    }

    // ───────────────────────── 악마 거래 ─────────────────────────

    fn devil_prepare(&mut self, chest: TileId) {
        let mut offer: Option<DevilOffer> = None;
        // 1) 포탑 제안
        if self.rng_v.float() < 0.2 {
            let mut a = 1usize;
            while a <= COLS && ((a != 1 && a != 6) || self.turrets[a].is_some()) {
                a += 1;
            }
            if a <= COLS {
                if self.rng_v.float() < 0.5 {
                    // 인벤토리 분기: 인벤토리가 비어 있으므로 제안 없음
                } else {
                    let towers: Vec<TileId> = self
                        .ak
                        .iter()
                        .copied()
                        .filter(|&t| self.tiles[t].kind == Kind::ArrowTower && self.tiles[t].gy > 0)
                        .collect();
                    if !towers.is_empty() {
                        let t = towers[self.rng_v.item_index(towers.len())];
                        offer = Some(DevilOffer::Turret { col: a as i32, tower: t });
                    }
                }
            }
        }
        if offer.is_none() {
            // 2) 대포 슬롯
            if self.rng_v.float() < 0.2 {
                let mut cands = Vec::new();
                for x in 1..=COLS as i32 {
                    let mut y = ROWS as i32;
                    while y > 1 {
                        if !self.is_slot(x, y) && self.is_ground(x, y) {
                            if let Some(s) = self.tile_at(x, y) {
                                let k = self.tiles[s].kind;
                                if k.is_resource() || k == Kind::Cannon || s == chest {
                                    cands.push(s);
                                }
                            }
                            break;
                        }
                        y -= 1;
                    }
                }
                if !cands.is_empty() {
                    let s = cands[self.rng_v.int(1, cands.len() as i64) as usize - 1];
                    offer = Some(DevilOffer::Slot { tile: s });
                }
            }
            // 3) 빙산
            if offer.is_none() && self.rng_v.float() < 0.2 {
                if let Some((x, y)) = self.pick_iceberg_cell() {
                    let tier = self.rng_v.int(1, 3) as u8;
                    offer = Some(DevilOffer::Iceberg { x, y, tier });
                }
            }
            // 4) 방어물 제안
            if offer.is_none() {
                let kind = self.random_final_kind(None);
                let tier = if self.deals_done > 0 || self.deals_declined > 0 { 2 } else { 1 };
                let mut tm = chest;
                let mut best = 0.0f64;
                for y in 1..=ROWS as i32 {
                    for x in 1..=COLS as i32 {
                        if let Some(i) = self.tile_at(x, y) {
                            let ik = self.tiles[i].kind;
                            if ik.upgraded().is_some() && ik.upgraded() != Some(ik) && self.can_go_here(kind, x, y) {
                                let mut t = self.rng_v.int(88, 100) as f64;
                                t /= (y + 3) as f64;
                                t /= self.neighbor_count(x, y, kind) as f64;
                                t *= self.neighbor_count(x, y, ik) as f64;
                                if t > best {
                                    best = t;
                                    tm = i;
                                }
                            }
                        }
                    }
                }
                offer = Some(DevilOffer::Place { kind, tier, tile: tm });
            }
        }
        // 가격: 하트가 2 이상일 때만 난수를 뽑는다
        let mut price = 0;
        if self.hearts > 1 && self.rng_v.float() > 0.5 {
            price = self.rng_v.int(2, 5) as i32 + self.deals_done.clamp(0, 4);
            price = price.clamp(0, self.hearts - 1);
        }
        self.pending.devil = Some((offer.unwrap(), price));
        self.phase = Phase::Devil;
    }

    /// 악마 거래 응답. 가격 0(원래 광고 조건)도 무료로 수락 가능.
    pub fn devil_answer(&mut self, accept: bool) -> ActionResult {
        if self.phase != Phase::Devil {
            return ActionResult::Invalid;
        }
        let (offer, price) = self.pending.devil.take().unwrap();
        if accept {
            self.deals_done += 1;
            self.hearts -= price;
            match offer {
                DevilOffer::Turret { col, tower } => {
                    self.turrets[col as usize] = Some(tower);
                    self.remove_delete_and_replace(tower, false);
                    self.tiles[tower].gy = 0;
                    self.tiles[tower].gx = col;
                }
                DevilOffer::Slot { tile } => {
                    let (x, y) = (self.tiles[tile].gx, self.tiles[tile].gy);
                    self.terrain[x as usize][y as usize] = Terrain::CannonSlot;
                    if self.tiles[tile].kind != Kind::Cannon {
                        self.remove_delete_and_replace(tile, false);
                    }
                    self.tiles[tile].removed = false;
                }
                DevilOffer::Iceberg { x, y, tier } => {
                    let id = self.new_tile(Kind::Iceberg);
                    let t = &mut self.tiles[id];
                    t.gx = x;
                    t.gy = y;
                    t.tier = tier;
                    self.set_grid(x, y, Some(id));
                }
                DevilOffer::Place { kind, tier, tile } => {
                    if kind == Kind::Chest {
                        self.chest_made[tier.min(4) as usize] += 1;
                        self.tiles[tile].born = (self.stat_steps, self.day);
                    }
                    let t = &mut self.tiles[tile];
                    t.kind = kind;
                    t.frame = 1;
                    t.tier = tier;
                    t.removed = false;
                }
            }
        } else {
            self.deals_declined += 1;
        }
        self.opened_chest = None;
        self.phase = Phase::Match;
        self.advance();
        ActionResult::Ok
    }

    // ───────────────────────── 상인 ─────────────────────────

    /// 상인에게 팔 타일 선택 (거절 불가). 여는 중인 상자는 팔 수 없다.
    pub fn merchant_sell(&mut self, x: i32, y: i32) -> ActionResult {
        if self.phase != Phase::Merchant {
            return ActionResult::Invalid;
        }
        let Some(t) = self.tile_at(x, y) else { return ActionResult::Invalid };
        if Some(t) == self.opened_chest {
            return ActionResult::Invalid;
        }
        let v = self.tile_value(t);
        self.add_swaps(v);
        self.remove_delete_and_replace(t, false);
        self.opened_chest = None;
        self.phase = Phase::Match;
        self.advance();
        ActionResult::Ok
    }

    // ───────────────────────── 상점 ─────────────────────────

    fn shop_prepare(&mut self) {
        let free = self.rng_v.int(1, 3);
        let mut items = [ShopItem::TntBlast; 3];
        let mut first_kind = Kind::Anvil;
        for s in 1..=3 {
            let _cost = self.rng_v.int(3, 20); // 골드 가격: 쓰지 않지만 난수는 소비한다
            let item = match s {
                1 => {
                    first_kind = self.random_final_kind(None);
                    ShopItem::Tile { kind: first_kind, tier: 1 }
                }
                2 => {
                    let mut h = self.rng_v.int(2, 4) as i32;
                    if self.hearts + h > MAX_HEARTS {
                        h = MAX_HEARTS - self.hearts;
                    }
                    if h <= 0 {
                        ShopItem::Tile { kind: self.random_final_kind(Some(first_kind)), tier: 2 }
                    } else {
                        ShopItem::Hearts(h)
                    }
                }
                _ => match self.rng_v.int(1, 4) {
                    1 => ShopItem::Tile { kind: Kind::Anvil, tier: self.rng_v.int(2, 3) as u8 },
                    2 => {
                        if self.rng_v.int(1, 20) == 1 {
                            ShopItem::Tile { kind: Kind::FairyHouse, tier: 1 }
                        } else {
                            ShopItem::Tile { kind: Kind::Fairy, tier: 0 }
                        }
                    }
                    3 => {
                        if self.pick_iceberg_cell().is_some() {
                            ShopItem::Tile { kind: Kind::Iceberg, tier: self.rng_v.int(1, 2) as u8 }
                        } else {
                            ShopItem::TntBlast
                        }
                    }
                    _ => ShopItem::TntBlast,
                },
            };
            items[s - 1] = item;
        }
        self.pending.shop = Some(items[free as usize - 1]);
    }

    /// 상점: 무료 품목을 받을지(참) 거절할지(거짓)
    pub fn shop_answer(&mut self, take: bool) -> ActionResult {
        if self.phase != Phase::Shop {
            return ActionResult::Invalid;
        }
        let item = self.pending.shop.take().unwrap();
        if !take {
            self.opened_chest = None;
            self.phase = Phase::Match;
            self.advance();
            return ActionResult::Ok;
        }
        match item {
            ShopItem::Hearts(h) => {
                if self.hearts < MAX_HEARTS {
                    self.hearts = (self.hearts + h).min(MAX_HEARTS);
                }
                self.opened_chest = None;
                self.phase = Phase::Match;
                self.advance();
            }
            ShopItem::TntBlast => {
                self.pending.tnt_source = None;
                self.pending.tnt_cancelable = false;
                self.phase = Phase::TntMenu;
            }
            ShopItem::Tile { kind, tier } => {
                self.pending.place = Some((kind, tier));
                self.phase = Phase::PlaceItem;
            }
        }
        ActionResult::Ok
    }

    /// `ui(kind, tier, x, y)`: 아이템·요정이 빈 곳/0행에 놓일 수 있는지
    pub(crate) fn can_place(&self, kind: Kind, tier: u8, x: i32, y: i32) -> bool {
        if y == 0 {
            (kind == Kind::ArrowTower && (x == 1 || x == 6) && self.turret_slot_open(x, tier))
                || (kind == Kind::Stone && self.hearts < MAX_HEARTS && !self.closed_day())
        } else {
            self.can_go_here(kind, x, y)
        }
    }

    /// `mH`: 포탑 슬롯이 비었거나 더 낮은 등급
    fn turret_slot_open(&self, x: i32, tier: u8) -> bool {
        match self.turrets[x as usize] {
            None => true,
            Some(o) => self.tiles[o].tier < tier,
        }
    }

    /// 상점 품목을 놓을 칸 선택 (`_9`): 기존 타일이 있으면 덮어쓴다
    pub fn place_item(&mut self, x: i32, y: i32) -> ActionResult {
        if self.phase != Phase::PlaceItem {
            return ActionResult::Invalid;
        }
        let (kind, tier) = self.pending.place.unwrap();
        if !(1..=COLS as i32).contains(&x) || !(0..=ROWS as i32).contains(&y) || !self.can_place(kind, tier, x, y) {
            return ActionResult::Invalid;
        }
        self.pending.place = None;
        let id = match if y >= 1 { self.tile_at(x, y) } else { None } {
            Some(s) => s,
            None => {
                let s = self.new_tile(kind);
                self.tiles[s].gx = x;
                if y < 1 {
                    // 원본: 포탑으로 놓은 타일도 타일 목록에 남는다
                    self.turrets[x as usize] = Some(s);
                } else {
                    self.tiles[s].gy = y;
                    self.set_grid(x, y, Some(s));
                }
                s
            }
        };
        if kind == Kind::Chest {
            self.chest_made[tier.min(4) as usize] += 1;
            self.tiles[id].born = (self.stat_steps, self.day);
        }
        let loops = self.loops;
        let t = &mut self.tiles[id];
        t.moved_time = loops;
        t.kind = kind;
        t.frame = 1;
        t.tier = tier;
        t.removed = false;
        t.reload = 0.0;
        if kind == Kind::Tnt {
            t.dynamite = 1;
        }
        if kind == Kind::FairyHouse {
            t.frame = 2;
        }
        self.opened_chest = None;
        self.phase = Phase::Match;
        self.advance();
        ActionResult::Ok
    }

    // ───────────────────────── 지니 ─────────────────────────

    /// 지니: 자원 종류를 고르면 다른 종류의 자원 최대 3개를 그 종류로 바꾼다
    pub fn genie_pick(&mut self, kind: Kind) -> ActionResult {
        if self.phase != Phase::Genie || !kind.is_resource() {
            return ActionResult::Invalid;
        }
        'outer: for _ in 0..3 {
            let mut tries = 0;
            loop {
                let x = self.rng_v.int(1, COLS as i64) as i32;
                let y = self.rng_v.int(1, ROWS as i64) as i32;
                if let Some(t) = self.tile_at(x, y) {
                    let k = self.tiles[t].kind;
                    if k.is_resource() && k != kind {
                        self.tiles[t].kind = kind;
                        self.tiles[t].frame = 1;
                        break;
                    }
                }
                tries += 1;
                if tries > 3333 {
                    break 'outer;
                }
            }
        }
        self.opened_chest = None;
        self.phase = Phase::Match;
        self.advance();
        ActionResult::Ok
    }

    // ───────────────────────── TNT ─────────────────────────

    fn tnt_finish(&mut self) {
        if let Some(s) = self.pending.tnt_source.take() {
            if self.tiles[s].dynamite <= 0 {
                self.tiles[s].removed = true;
            }
        }
        self.opened_chest = None;
        self.phase = Phase::Match;
        self.advance();
    }

    /// 보드의 모든 자원(T0)을 폭파
    pub fn tnt_blast_all(&mut self) -> ActionResult {
        if self.phase != Phase::TntMenu {
            return ActionResult::Invalid;
        }
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if let Some(t) = self.tile_at(x, y) {
                    if self.tiles[t].kind.is_resource() {
                        self.tiles[t].removed = true;
                    }
                }
            }
        }
        if let Some(s) = self.pending.tnt_source {
            self.tiles[s].dynamite -= 1;
        }
        self.tnt_finish();
        ActionResult::Ok
    }

    /// tier 0 타일 하나 폭파 (TNT 자신 제외)
    pub fn tnt_blast_one(&mut self, x: i32, y: i32) -> ActionResult {
        if self.phase != Phase::TntMenu {
            return ActionResult::Invalid;
        }
        let Some(t) = self.tile_at(x, y) else { return ActionResult::Invalid };
        if self.tiles[t].tier >= 1 || Some(t) == self.pending.tnt_source {
            return ActionResult::Invalid;
        }
        self.tiles[t].removed = true;
        if let Some(s) = self.pending.tnt_source {
            self.tiles[s].dynamite -= 1;
        }
        self.tnt_finish();
        ActionResult::Ok
    }

    /// TNT 창 닫기 (TNT 타일에서 연 경우만)
    pub fn tnt_cancel(&mut self) -> ActionResult {
        if self.phase != Phase::TntMenu || !self.pending.tnt_cancelable {
            return ActionResult::Invalid;
        }
        self.pending.tnt_source = None;
        self.phase = self.resume_phase;
        ActionResult::Ok
    }

    // ───────────────────────── 요정 ─────────────────────────

    fn fairy_start(&mut self, t: TileId) {
        self.resume_phase = self.phase;
        self.pending.fairy = Some(t);
        self.pending.fairy_src = None;
        self.phase = Phase::FairySource;
    }

    /// 요정이 집을 타일 (Fairy 타일은 집을 수 없다)
    pub fn fairy_pick_source(&mut self, x: i32, y: i32) -> ActionResult {
        if self.phase != Phase::FairySource {
            return ActionResult::Invalid;
        }
        let Some(t) = self.tile_at(x, y) else { return ActionResult::Invalid };
        if self.tiles[t].kind == Kind::Fairy {
            return ActionResult::Invalid;
        }
        self.pending.fairy_src = Some(t);
        self.phase = Phase::FairyTarget;
        ActionResult::Ok
    }

    /// 요정이 놓을 곳: 타일이면 교환, 빈 곳이면 이동(0행은 포탑 장착·성 보수)
    pub fn fairy_pick_target(&mut self, x: i32, y: i32) -> ActionResult {
        if self.phase != Phase::FairyTarget || !(1..=COLS as i32).contains(&x) || !(0..=ROWS as i32).contains(&y) {
            return ActionResult::Invalid;
        }
        let src = self.pending.fairy_src.unwrap();
        let fairy = self.pending.fairy.unwrap();
        let target = if y >= 1 { self.tile_at(x, y) } else { None };
        if let Some(z) = target {
            if !self.swap_valid(src, z) {
                return ActionResult::Invalid;
            }
            self.loops += 1000;
            let (ax, ay, bx, by) = (self.tiles[src].gx, self.tiles[src].gy, self.tiles[z].gx, self.tiles[z].gy);
            self.set_grid(bx, by, Some(src));
            self.set_grid(ax, ay, Some(z));
            self.tiles[src].gx = bx;
            self.tiles[src].gy = by;
            self.tiles[z].gx = ax;
            self.tiles[z].gy = ay;
            self.tiles[src].moved_time = self.loops + 1;
            self.tiles[z].moved_time = self.loops;
            // 원본: 교환 직후 다음 프레임에 요정 확인·제거가 일어나 그때 생긴 타일의 시각은 교환 시각을 넘지 않는다
            self.fairy_swap_time = Some(self.loops);
        } else {
            let (kind, tier) = (self.tiles[src].kind, self.tiles[src].tier);
            if !self.can_place(kind, tier, x, y) {
                return ActionResult::Invalid;
            }
            self.loops += 1000;
            if y < 1 {
                if kind == Kind::ArrowTower && (x == 1 || x == 6) && self.turret_slot_open(x, tier) {
                    self.remove_delete_and_replace(src, false);
                    self.turrets[x as usize] = Some(src);
                    self.tiles[src].gx = x;
                    self.tiles[src].gy = 0;
                } else {
                    self.remove_delete_and_replace(src, true);
                    self.hearts = (self.hearts + 1).min(MAX_HEARTS);
                }
            } else {
                self.move_into_empty(src, x, y);
            }
        }
        if self.tiles[fairy].kind == Kind::FairyHouse {
            self.tiles[fairy].frame = (self.tiles[fairy].frame - 1).max(1);
        } else {
            self.fairy_in_use = Some(fairy);
            self.fairy_check_did = false;
        }
        self.pending.fairy = None;
        self.pending.fairy_src = None;
        self.phase = Phase::Fall;
        self.advance();
        ActionResult::Ok
    }

    /// 요정 창 닫기 (놓을 곳을 고르기 전까지)
    pub fn fairy_cancel(&mut self) -> ActionResult {
        if !matches!(self.phase, Phase::FairySource | Phase::FairyTarget) {
            return ActionResult::Invalid;
        }
        self.pending.fairy = None;
        self.pending.fairy_src = None;
        self.phase = self.resume_phase;
        ActionResult::Ok
    }
}
