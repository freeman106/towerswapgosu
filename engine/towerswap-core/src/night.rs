//! 밤(공격) 단계와 하루 전환. 원본 m7(웨이브), mD(드래곤), ua.update6/shootEnemy(타워),
//! mL(투사체), mx/m4/mw(하루 전환)를 프레임 단위로 재현한다.

use crate::game::{ActionResult, Game, Phase, TileId};
use crate::jsmath::{angle_to, cos, sin, DEG2RAD};
use crate::kinds::Kind;
use crate::level::{COLS, ROWS};

const PROJ_SPEED: f64 = 3.84;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragonKind {
    Green,  // tC
    Red,    // t5
    Mother, // tT
}

impl DragonKind {
    /// 넉백 계수 `lightweight`
    fn weight(self) -> f64 {
        match self {
            DragonKind::Green => 3.0,
            DragonKind::Red => 2.0,
            DragonKind::Mother => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dragon {
    pub kind: DragonKind,
    pub px: f64,
    pub py: f64,
    pub gx: i32,
    pub gy: i32,
    pub health: f64,
    pub health_max: f64,
    pub speed: f64,
    pub size: f64,
    pub is_dead: bool,
    pub dead_time: i64,
    pub got_to_castle: bool,
    pub spawn_loop: i64,
    pub arrows_coming: i32,
    pub baby_angle: f64,
    pub baby_loops: f64,
}

#[derive(Clone, Debug)]
pub struct Projectile {
    pub tower: TileId,
    pub tower_kind: Kind,
    pub damage: f64,
    pub px: f64,
    pub py: f64,
    pub target: Option<usize>,
    pub tx: f64,
    pub ty: f64,
    pub dir: f64,
    pub shot_up: bool,
    pub exploded: i32,
}

/// `ea.rectshit`: 엄격 부등호 겹침
#[inline]
fn rects_hit(e: f64, a: f64, i: f64, t: f64, s: f64, r: f64, n: f64, o: f64) -> bool {
    !(e >= s + n || a >= r + o || s >= e + i || r >= a + t)
}

/// `ea.mod`
#[inline]
fn js_mod(e: f64, a: f64) -> f64 {
    if e > 0.0 {
        e % a
    } else if a > 0.0 {
        e - a * (e / a).floor()
    } else if a == 0.0 {
        e
    } else {
        e % a
    }
}

/// 조준 격자의 행 범위: gy = GY0..=9 (gy >= 10은 사격 불가)
const GY0: i32 = -2;
const NGY: usize = (10 - GY0) as usize;
const NCELL: usize = COLS * NGY;

#[inline]
fn target_cell(gx: i32, gy: i32) -> usize {
    (gx - 1) as usize * NGY + (gy - GY0) as usize
}

#[derive(Clone, Default)]
pub struct NightState {
    pub dragons: Vec<Dragon>,
    pub a4: Vec<usize>, // 살아 있는(목록에 있는) 드래곤, 생성 순서
    pub projectiles: Vec<Projectile>,
    pub a6: Vec<usize>,
    pub ne: i64, // 원본 nE (상태 0,1,2,3,25에서 프레임마다 증가)
    /// 사격하는 타일: 포탑(1..6열) 다음 타일 목록 순서. 밤 동안 타일은 바뀌지 않는다.
    towers: Vec<TileId>,
    /// 프레임마다 만드는 조준 격자: 사격 가능한 드래곤을 칸(열 우선)별로 a4 순서를 유지해 모은다.
    /// 칸 c의 드래곤은 grid_ids[grid_start[c]..grid_start[c + 1]], 값은 (a4 순번, 드래곤).
    grid_start: Vec<u32>,
    grid_ids: Vec<(u32, u32)>,
    /// 격자 범위 밖(열 1..6, 행 GY0..9 밖)의 사격 가능한 드래곤. 정상 진행에서는 비어 있다.
    grid_extra: Vec<(u32, u32)>,
    grid_tmp: Vec<u16>,
    /// 칸별 감속 (감속 구간 시작 비율, 곱할 값): 얼음벽·빙산 칸. 밤 시작 때 만든다.
    slow: [[Option<(f64, f64)>; ROWS + 1]; COLS + 1],
}

/// 드래곤 갱신 결과
enum DragonUpd {
    Keep,
    Remove,
    GameOver,
}

impl Game {
    // ───────────────────────── 웨이브 생성 ─────────────────────────

    /// `$q()`: 적 열 추첨 (First는 모든 열 유효)
    fn random_col(&mut self) -> i32 {
        self.rng_m.int(1, COLS as i64) as i32
    }

    /// `goalNumberNextGet`
    pub fn goal_next(&self) -> i32 {
        match self.achievements {
            0 => 10,
            1 => 20,
            2 => 30,
            3 => 40,
            4 => 50,
            _ => 0,
        }
    }

    /// `new mD`: 생성자에서 M 난수 1개(애니메이션 프레임)를 쓴다
    fn new_dragon(&mut self) -> usize {
        let _frame = 2.0 * self.rng_m.float() + 1.0;
        let d = Dragon {
            kind: DragonKind::Green,
            px: 0.0,
            py: 432.0,
            gx: 0,
            gy: (432.0f64 / 48.0).floor() as i32 + 1,
            health: 0.0,
            health_max: 0.0,
            speed: 0.0,
            size: 0.0,
            is_dead: false,
            dead_time: 0,
            got_to_castle: false,
            spawn_loop: 0,
            arrows_coming: 0,
            baby_angle: 0.0,
            baby_loops: 0.0,
        };
        let id = self.night.dragons.len();
        self.night.dragons.push(d);
        self.night.a4.push(id);
        id
    }

    /// `mD.prototype.kindSet`
    fn dragon_kind_set(&mut self, id: usize, kind: DragonKind) {
        let tr = self.day as f64;
        match kind {
            DragonKind::Green => {
                let gx = self.random_col();
                let jitter = (self.rng_m.float() - 0.5) * 4.800000000000001;
                let d = &mut self.night.dragons[id];
                d.health = 6.0;
                d.size = 38.0;
                d.speed = 0.7 * (1.0 + 0.013 * tr);
                d.gx = gx;
                d.px = (gx as f64 - 0.5) * 48.0 + jitter;
            }
            DragonKind::Red => {
                let gx = self.random_col();
                let jitter = (self.rng_m.float() - 0.5) * 2.4000000000000004;
                let d = &mut self.night.dragons[id];
                d.health = 12.0;
                d.size = 38.0;
                d.speed = 0.7 * (1.0 + 0.01 * tr);
                d.gx = gx;
                d.px = (gx as f64 - 0.5) * 48.0 + jitter;
            }
            DragonKind::Mother => {
                let rw = self.boss_col;
                let d = &mut self.night.dragons[id];
                d.health = -13.0 + 11.0 * tr;
                d.speed = 0.6 * (1.0 + 0.01 * tr);
                if tr > 40.0 {
                    d.health += 99.0;
                    d.speed += 0.3 * d.speed;
                }
                d.size = 0.7 * 74.0;
                d.gx = rw;
                d.px = (rw as f64 - 0.5) * 48.0;
            }
        }
        let d = &mut self.night.dragons[id];
        d.kind = kind;
        d.health_max = d.health;
    }

    /// `m7()`: 웨이브 생성
    fn make_wave(&mut self) {
        let tr = self.day as f64;
        let mut s = 48.0 * (tr / 14.0 + 1.0);
        let mut e = (2.6 * tr).ceil() + 2.0;
        if tr > 50.0 {
            e = (0.4 * e).ceil() - 10.0;
            s = (0.8 * s).ceil();
            e += (tr - 50.0) * 13.0;
        }
        e -= 1.0; // First 레벨
        if e > 0.0 {
            let mut i = 1.0;
            while i <= e {
                let a = self.new_dragon();
                if i == 1.0 && self.day == self.goal_next() {
                    self.dragon_kind_set(a, DragonKind::Mother);
                    self.night.dragons[a].spawn_loop = self.night.ne;
                    self.night.dragons[a].py += 44.0;
                    break;
                }
                if self.day > 50 {
                    self.dragon_kind_set(a, DragonKind::Red);
                } else {
                    self.dragon_kind_set(a, DragonKind::Green);
                }
                let d = &mut self.night.dragons[a];
                d.py += i / e * s;
                d.gy = (d.py / 48.0).floor() as i32 + 1;
                i += 1.0;
            }
        }
        self.phase = Phase::Night;
    }

    // ───────────────────────── 드래곤 ─────────────────────────

    fn dragon_grid_x_set(&mut self, id: usize) {
        let d = &mut self.night.dragons[id];
        d.gx = ((d.px / 48.0).floor() as i32 + 1).clamp(1, COLS as i32);
    }

    /// `speedNowGet`
    fn dragon_speed_now(&self, id: usize) -> f64 {
        let d = &self.night.dragons[id];
        if d.is_dead {
            return 0.0;
        }
        let mut e = d.speed;
        if Self::in_board(d.gx, d.gy) {
            if let Some((i, m)) = self.night.slow[d.gx as usize][d.gy as usize] {
                let f = js_mod(d.py / 48.0, 1.0);
                if f > i && f < i + 0.5 {
                    e *= m;
                }
            }
        }
        let el = self.night.ne - d.spawn_loop;
        if el < 50 {
            e *= el as f64 / 50.0;
        }
        e
    }

    /// `die()`: 마더면 아기 드래곤을 낳는다
    fn dragon_die(&mut self, id: usize) {
        if self.night.dragons[id].dead_time == 0 && self.night.dragons[id].kind == DragonKind::Mother {
            let (mx, my) = (self.night.dragons[id].px, self.night.dragons[id].py);
            let tr = self.day as f64;
            let n = 0.7 * tr + 3.0;
            let mut t = 1.0;
            while t <= n {
                let b = self.new_dragon();
                self.dragon_kind_set(b, DragonKind::Green);
                let ne = self.night.ne;
                let angle = self.rng_m.int(0, 360) as f64;
                let loops = self.rng_m.int_f(5.0, 20.0 + 0.6 * tr);
                let d = &mut self.night.dragons[b];
                d.spawn_loop = ne;
                d.px = mx;
                d.py = my;
                d.baby_angle = angle;
                d.baby_loops = loops;
                self.dragon_grid_x_set(b);
                t += 1.0;
            }
        }
        let loops = self.loops;
        let d = &mut self.night.dragons[id];
        d.is_dead = true;
        d.dead_time = loops;
        d.size *= 0.6;
    }

    /// `damageTheCastle`: 하트가 0이 되면 게임 오버, 아니면 드래곤을 목록에서 뺀다
    fn damage_castle(&mut self) -> DragonUpd {
        self.hearts -= 1;
        if self.hearts <= 0 {
            self.phase = Phase::GameOver;
            return DragonUpd::GameOver;
        }
        DragonUpd::Remove
    }

    /// `mD.prototype.update2`
    fn dragon_update(&mut self, id: usize) -> DragonUpd {
        if self.night.dragons[id].is_dead {
            let d = &self.night.dragons[id];
            if d.got_to_castle {
                if self.loops - d.dead_time > 15 {
                    return self.damage_castle();
                }
            } else {
                let d = &mut self.night.dragons[id];
                d.py += 0.3 * d.speed;
            }
            if self.loops - self.night.dragons[id].dead_time > 20 {
                return DragonUpd::Remove;
            }
            return DragonUpd::Keep;
        }
        if self.night.dragons[id].baby_loops > 0.0 {
            let d = &mut self.night.dragons[id];
            d.baby_loops -= 1.0;
            d.px += cos(d.baby_angle * DEG2RAD);
            d.py += sin(d.baby_angle * DEG2RAD);
            self.dragon_grid_x_set(id);
        }
        // direction 3: 위로
        let v = self.dragon_speed_now(id);
        self.night.dragons[id].py -= v;
        self.dragon_grid_x_set(id);
        let d = &mut self.night.dragons[id];
        if d.px < 0.0 {
            d.px += 0.5;
        }
        if d.px > (COLS as f64) * 48.0 {
            d.px -= 0.5;
        }
        d.gy = (d.py / 48.0).floor() as i32 + 1;
        if d.py < -48.0 {
            d.got_to_castle = true;
            d.is_dead = true;
            d.dead_time = self.loops;
            return DragonUpd::Keep;
        }
        if d.kind == DragonKind::Mother && d.py < 0.0 {
            self.dragon_die(id);
        }
        DragonUpd::Keep
    }

    /// `takeDamage`
    fn dragon_take_damage(&mut self, id: usize, p: usize) {
        if self.night.dragons[id].is_dead {
            return;
        }
        let (dmg, kind, dir) = {
            let pr = &self.night.projectiles[p];
            (pr.damage, pr.tower_kind, pr.dir)
        };
        self.night.dragons[id].health -= dmg;
        if self.night.dragons[id].health <= 0.0 {
            self.dragon_die(id);
        } else {
            let d = &mut self.night.dragons[id];
            if kind == Kind::ArrowTower {
                d.arrows_coming -= 1;
            }
            let w = d.kind.weight();
            d.px += cos(dir * DEG2RAD) * w;
            d.py += sin(dir * DEG2RAD) * (w / 2.0);
        }
    }

    // ───────────────────────── 타워 ─────────────────────────

    fn tile_px(&self, t: TileId) -> (f64, f64) {
        let tile = &self.tiles[t];
        (((tile.gx - 1) * 48) as f64, ((tile.gy - 1) * 48) as f64)
    }

    fn is_turret(&self, t: TileId) -> bool {
        self.turrets[1] == Some(t) || self.turrets[6] == Some(t)
    }

    /// `shootEnemyCan`
    fn can_shoot(&self, t: TileId, e: usize) -> bool {
        let d = &self.night.dragons[e];
        if d.is_dead || d.gy >= 10 {
            return false;
        }
        let tile = &self.tiles[t];
        match tile.kind {
            Kind::Ballista => tile.gx == d.gx,
            Kind::Cannon => tile.gy == d.gy,
            Kind::ArrowTower => {
                let r = if self.is_turret(t) { 2 } else { 1 };
                r >= (tile.gx - d.gx).abs() && r >= (tile.gy - d.gy).abs()
            }
            _ => false,
        }
    }

    /// `shootEnemyValueGet`
    fn shoot_value(&self, t: TileId, e: usize) -> f64 {
        let d = &self.night.dragons[e];
        let tile = &self.tiles[t];
        let (px, py) = self.tile_px(t);
        match tile.kind {
            Kind::Ballista => 48.0 / ((py - d.py).abs() + 48.0),
            Kind::ArrowTower => {
                let mut s = 288.0 - ((px + 24.0 - d.px).abs() + (py + 24.0 - d.py).abs());
                if d.arrows_coming as f64 * 3.0 >= d.health {
                    s *= 0.1;
                }
                if d.arrows_coming == 0 && d.health < d.health_max {
                    s *= 2.0;
                }
                s
            }
            Kind::Cannon => {
                let mut i = px + 24.0;
                if tile.flipped_preferred {
                    i -= 26.400000000000002;
                } else {
                    i += 26.400000000000002;
                }
                let mut a = 0.0;
                if (d.gx < tile.gx) == tile.flipped_preferred || tile.gx == d.gx {
                    a = 3168.0;
                }
                if d.gx > 6 {
                    a = 4752.0;
                }
                (48.0 + a) / ((i - d.px).abs() + 48.0)
            }
            _ => 1.0,
        }
    }

    /// `shootEnemy`
    fn shoot(&mut self, t: TileId, e: usize) {
        let (kind, tier, gx, gy) = {
            let tile = &self.tiles[t];
            (tile.kind, tile.tier, tile.gx, tile.gy)
        };
        let (dmg, reload) = kind.weapon().unwrap();
        let mut rd = reload / 2.4f64.powf(tier as f64 - 1.0);
        if kind == Kind::Cannon {
            if self.terrain_at(gx, gy) == crate::level::Terrain::CannonSlot {
                rd *= 0.7;
            }
            let egx = self.night.dragons[e].gx;
            if egx != gx {
                self.tiles[t].flipped = egx < gx;
            }
        }
        self.tiles[t].reload = rd;
        let (tpx, tpy) = self.tile_px(t);
        let mut p = Projectile {
            tower: t,
            tower_kind: kind,
            damage: dmg,
            px: tpx + 24.0,
            py: tpy + 24.0,
            target: Some(e),
            tx: 0.0,
            ty: 0.0,
            dir: 0.0,
            shot_up: false,
            exploded: 0,
        };
        if gy <= 0 {
            p.py -= 3.0;
        }
        let (epx, epy) = (self.night.dragons[e].px, self.night.dragons[e].py);
        match kind {
            Kind::ArrowTower => self.night.dragons[e].arrows_coming += 1,
            Kind::Ballista => {
                p.shot_up = epy < tpy;
                p.tx = epx;
                p.ty = epy;
            }
            Kind::Cannon => {
                p.tx = epx;
                p.ty = epy;
                let sp = self.dragon_speed_now(e).abs();
                p.ty -= (p.px - p.tx).abs() / PROJ_SPEED * sp;
                p.target = None;
                if self.tiles[t].flipped {
                    p.dir += 180.0;
                    p.px += -12.0;
                } else {
                    p.px += 12.0;
                }
                p.py -= 1.0;
            }
            _ => {}
        }
        let id = self.night.projectiles.len();
        self.night.projectiles.push(p);
        self.night.a6.push(id);
    }

    /// 조준 격자를 다시 만든다 (타워 사격 직전, 프레임마다). 칸별 계수 정렬이라 칸 안에서는 a4 순서가 유지된다.
    fn build_target_grid(&mut self) {
        const SKIP: u16 = NCELL as u16;
        let n = &mut self.night;
        let mut start = [0u32; NCELL + 1];
        n.grid_extra.clear();
        n.grid_tmp.clear();
        for (i, &e) in n.a4.iter().enumerate() {
            let d = &n.dragons[e];
            let mut c = SKIP;
            if !d.is_dead && d.gy < 10 {
                if (1..=COLS as i32).contains(&d.gx) && d.gy >= GY0 {
                    c = target_cell(d.gx, d.gy) as u16;
                    start[c as usize + 1] += 1;
                } else {
                    n.grid_extra.push((i as u32, e as u32));
                }
            }
            n.grid_tmp.push(c);
        }
        for c in 0..NCELL {
            start[c + 1] += start[c];
        }
        n.grid_ids.resize(start[NCELL] as usize, (0, 0));
        let mut pos = start;
        for (i, &c) in n.grid_tmp.iter().enumerate() {
            if c != SKIP {
                let c = c as usize;
                n.grid_ids[pos[c] as usize] = (i as u32, n.a4[i] as u32);
                pos[c] += 1;
            }
        }
        n.grid_start.clear();
        n.grid_start.extend_from_slice(&start);
    }

    /// 후보 중 최고 가치의 드래곤. 원본은 a4를 앞에서부터 훑으며 더 큰 값만 받으므로, 동점이면 a4 순번이 앞선 쪽이다.
    fn pick_target(&self, t: TileId, cands: &[(u32, u32)], best: &mut Option<(u32, usize, f64)>) {
        for &(order, e) in cands {
            let e = e as usize;
            if self.can_shoot(t, e) {
                let v = self.shoot_value(t, e);
                if best.map_or(true, |(bo, _, bv)| v > bv || (v == bv && order < bo)) {
                    *best = Some((order, e, v));
                }
            }
        }
    }

    /// 격자 칸 c0..c1(열 우선 번호)의 후보
    #[inline]
    fn grid_range(&self, c0: usize, c1: usize) -> &[(u32, u32)] {
        let s = &self.night.grid_start;
        &self.night.grid_ids[s[c0] as usize..s[c1] as usize]
    }

    /// `ua.prototype.update6` (밤). 사거리 안의 격자 칸만 훑는다.
    fn tower_update(&mut self, t: TileId) {
        if self.tiles[t].reload <= 0.0 {
            let (kind, tx, ty) = (self.tiles[t].kind, self.tiles[t].gx, self.tiles[t].gy);
            let mut best = None;
            let cols = 1..=COLS as i32;
            match kind {
                Kind::Ballista if cols.contains(&tx) => {
                    let c = target_cell(tx, GY0);
                    self.pick_target(t, self.grid_range(c, c + NGY), &mut best);
                }
                Kind::Cannon if (GY0..10).contains(&ty) => {
                    for x in cols {
                        let c = target_cell(x, ty);
                        self.pick_target(t, self.grid_range(c, c + 1), &mut best);
                    }
                }
                Kind::ArrowTower => {
                    let r = if self.is_turret(t) { 2 } else { 1 };
                    let (y0, y1) = ((ty - r).max(GY0), (ty + r).min(9));
                    if y0 <= y1 {
                        for x in (tx - r).max(1)..=(tx + r).min(COLS as i32) {
                            self.pick_target(t, self.grid_range(target_cell(x, y0), target_cell(x, y1) + 1), &mut best);
                        }
                    }
                }
                _ => {}
            }
            self.pick_target(t, &self.night.grid_extra, &mut best);
            if let Some((_, e, _)) = best {
                self.shoot(t, e);
            }
        }
        self.tiles[t].reload -= 1.0;
    }

    // ───────────────────────── 투사체 ─────────────────────────

    /// `hitEnemy`
    fn proj_hits(&self, p: usize, e: usize) -> bool {
        let pr = &self.night.projectiles[p];
        let d = &self.night.dragons[e];
        let a = if pr.exploded > 0 { 15.0 } else { 2.0 };
        rects_hit(pr.px - a / 2.0, pr.py - a / 2.0, a, a, d.px - d.size / 3.0, d.py - d.size / 2.0, 0.66 * d.size, d.size)
    }

    /// `retarget`. 화살이 대상을 못 찾으면 투사체를 지운다(참 반환)
    fn proj_retarget(&mut self, p: usize) -> bool {
        let kind = self.night.projectiles[p].tower_kind;
        if kind == Kind::Ballista {
            let tgx = self.tiles[self.night.projectiles[p].tower].gx;
            for idx in 0..self.night.a4.len() {
                let e = self.night.a4[idx];
                let d = &self.night.dragons[e];
                let pr = &self.night.projectiles[p];
                if !d.is_dead && d.gx == tgx && (d.py < pr.py) == pr.shot_up {
                    let (x, y) = (d.px, d.py);
                    let pr = &mut self.night.projectiles[p];
                    pr.target = Some(e);
                    pr.tx = x;
                    pr.ty = y;
                    break;
                }
            }
            return false;
        }
        if kind == Kind::ArrowTower {
            for idx in 0..self.night.a4.len() {
                let e = self.night.a4[idx];
                let d = &self.night.dragons[e];
                let pr = &self.night.projectiles[p];
                if !d.is_dead && 9.0 > (pr.px - d.px).abs() && 9.0 > (pr.py - d.py).abs() {
                    self.night.projectiles[p].target = Some(e);
                    return false;
                }
            }
        }
        if kind == Kind::Cannon {
            self.night.projectiles[p].exploded = 1;
            false
        } else {
            true
        }
    }

    /// `mL.prototype.update3`. 투사체가 목록에 남으면 참
    fn proj_update(&mut self, p: usize) -> bool {
        if self.night.projectiles[p].exploded > 0 {
            self.night.projectiles[p].exploded += 1;
            if self.night.projectiles[p].exploded == 5 {
                for idx in 0..self.night.a4.len() {
                    let e = self.night.a4[idx];
                    if self.proj_hits(p, e) {
                        let (ppx, ppy) = (self.night.projectiles[p].px, self.night.projectiles[p].py);
                        let d = &mut self.night.dragons[e];
                        let ang = angle_to(ppx, ppy, d.px, d.py);
                        let w = d.kind.weight();
                        d.px += cos(ang * DEG2RAD) * w * 0.3;
                        d.py += sin(ang * DEG2RAD) * w * 0.3;
                    }
                }
            }
            return self.night.projectiles[p].exploded <= 15;
        }
        let kind = self.night.projectiles[p].tower_kind;
        if kind != Kind::Cannon {
            if let Some(t) = self.night.projectiles[p].target {
                if self.night.dragons[t].is_dead {
                    if self.proj_retarget(p) {
                        return false;
                    }
                    if kind != Kind::Ballista {
                        return true;
                    }
                }
                let t = self.night.projectiles[p].target.unwrap();
                let (dx, dy) = (self.night.dragons[t].px + 0.0, self.night.dragons[t].py + 0.0);
                let pr = &mut self.night.projectiles[p];
                pr.dir = angle_to(pr.px, pr.py, dx, dy);
            }
        }
        {
            let pr = &mut self.night.projectiles[p];
            if pr.target.is_none() {
                pr.dir = angle_to(pr.px, pr.py, pr.tx, pr.ty);
            }
            pr.px += cos(pr.dir * DEG2RAD) * PROJ_SPEED;
            pr.py += sin(pr.dir * DEG2RAD) * PROJ_SPEED;
        }
        if let Some(t) = self.night.projectiles[p].target {
            if self.proj_hits(p, t) {
                self.dragon_take_damage(t, p);
                if kind == Kind::Cannon {
                    self.night.projectiles[p].exploded = 1;
                } else {
                    return false;
                }
            }
        } else if kind != Kind::Ballista {
            let pr = &self.night.projectiles[p];
            if rects_hit(pr.px, pr.py, 5.0, 5.0, pr.tx, pr.ty, 5.0, 5.0) {
                for idx in 0..self.night.a4.len() {
                    let e = self.night.a4[idx];
                    if self.proj_hits(p, e) {
                        self.dragon_take_damage(e, p);
                        break;
                    }
                }
                self.night.projectiles[p].exploded = 1;
            }
        }
        true
    }

    // ───────────────────────── 밤 진행 ─────────────────────────

    /// 밤 직전(Dusk)에서 밤을 시작해 끝까지 진행하고, 다음 날 입력 대기 상태까지 간다.
    pub fn start_night(&mut self) {
        assert_eq!(self.phase, Phase::Dusk);
        // 상태 25: 30프레임 뒤 웨이브 생성 (그 사이 nE는 증가)
        self.night = Default::default();
        self.night.ne = self.ne_base;
        for _ in 0..30 {
            self.night.ne += 1;
            self.loops += 1;
        }
        self.night.ne += 1;
        self.make_wave();
        self.night_start_loops = self.loops;
        self.loops += 1;
        if self.night.a4.is_empty() {
            self.end_night();
            return;
        }
        // 사격하는 타일만 모은다(원본은 모든 타일을 돌지만 무기가 없으면 재장전 값만 줄고, 그 값은 밤이 끝나면 0이 된다).
        // 상점으로 놓은 포탑은 타일 목록에도 남아 있어 두 번 들어간다(원본과 같음).
        let mut towers: Vec<TileId> = self.turrets[1..=COLS].iter().flatten().copied().collect();
        towers.extend(self.ak.iter().copied());
        towers.retain(|&t| self.tiles[t].kind.weapon().is_some());
        self.night.towers = towers;
        for x in 1..=COLS {
            for y in 1..=ROWS {
                if let Some(t) = self.grid[x][y] {
                    let tile = &self.tiles[t];
                    if tile.kind == Kind::IceWall || tile.kind == Kind::Iceberg {
                        let i = if tile.kind == Kind::Iceberg { 0.5 } else { 0.1 };
                        self.night.slow[x][y] = Some((i, 1.0 - (0.42 + 0.12 * tile.tier as f64)));
                    }
                }
            }
        }
        loop {
            self.night.ne += 1;
            // 1) 포탑, 이어서 타일 목록 순서대로 사격
            self.build_target_grid();
            for i in 0..self.night.towers.len() {
                let t = self.night.towers[i];
                self.tower_update(t);
            }
            // 2) 드래곤 이동. 원본은 순회 중 현재 항목만 지우고 끝에 추가(아기 드래곤)할 수 있다.
            //    남는 항목을 앞으로 당기고 끝에서 자르면 순서가 같다.
            let (mut i, mut w) = (0, 0);
            while i < self.night.a4.len() {
                let d = self.night.a4[i];
                match self.dragon_update(d) {
                    DragonUpd::GameOver => return,
                    DragonUpd::Remove => {}
                    DragonUpd::Keep => {
                        self.night.a4[w] = d;
                        w += 1;
                    }
                }
                i += 1;
            }
            self.night.a4.truncate(w);
            // 3) 투사체 (같은 방식)
            let (mut i, mut w) = (0, 0);
            while i < self.night.a6.len() {
                let p = self.night.a6[i];
                if self.proj_update(p) {
                    self.night.a6[w] = p;
                    w += 1;
                }
                i += 1;
            }
            self.night.a6.truncate(w);
            if self.trace_night {
                let mut line = format!("F{}", self.loops - self.night_start_loops);
                for &d in &self.night.a4 {
                    let x = &self.night.dragons[d];
                    line += &format!(" {}:{:.6},{:.6},{},{}", d, x.px, x.py, x.health, x.is_dead as u8);
                }
                eprintln!("{line}");
                let mut pl = format!("P{}", self.loops - self.night_start_loops);
                for &p in &self.night.a6 {
                    let x = &self.night.projectiles[p];
                    pl += &format!(" {}:{:?}:{:.4},{:.4}:t{:?}:e{}:tx{:.4},{:.4}", p, x.tower_kind, x.px, x.py, x.target, x.exploded, x.tx, x.ty);
                }
                eprintln!("{pl}");
            }
            // 4) 밤 종료
            if self.night.a4.is_empty() && self.night.a6.is_empty() {
                self.end_night();
                return;
            }
            self.loops += 1;
            // 원본 그리기 함수(draw2)의 부작용: 죽은 드래곤(성 도달 제외)은 사망 후 20프레임 동안
            // 그릴 때마다 크기가 0.3씩 커진다. 커진 시체가 포탄 착탄 판정에 걸린다.
            for i in 0..self.night.a4.len() {
                let d = &mut self.night.dragons[self.night.a4[i]];
                if d.is_dead && !d.got_to_castle && self.loops - d.dead_time < 21 {
                    d.size += 0.3;
                }
            }
        }
    }

    /// 밤 종료 → day+1 → `mx()` → `m4()` → `mw()`
    fn end_night(&mut self) {
        self.ne_base = self.night.ne + 1000;
        self.day += 1;
        // mx: 보스 날이면 보스 열
        if self.day == self.goal_next() {
            self.boss_col = self.random_col();
        }
        // 업적
        if self.achievements < 5 && self.day == self.goal_next() + 1 {
            self.achievements = ((self.day - 1) / 10).min(5);
        }
        // 타워 재장전 초기화, 대포 현재 방향 = 선호 방향, 요정의 집 충전
        for y in 1..=ROWS as i32 {
            for x in 1..=COLS as i32 {
                if let Some(t) = self.tile_at(x, y) {
                    let tile = &mut self.tiles[t];
                    tile.reload = 0.0;
                    if tile.kind == Kind::Cannon {
                        tile.flipped = tile.flipped_preferred;
                    }
                    if tile.kind == Kind::FairyHouse {
                        tile.frame = tile.tier as i32 + 1;
                    }
                }
            }
        }
        for x in 1..=COLS {
            if let Some(t) = self.turrets[x] {
                self.tiles[t].reload = 0.0;
            }
        }
        self.morning_refill();
        self.loops += 1000;
        // mw
        self.phase = Phase::Fall;
        if self.day > 1 && self.day % 10 == 1 && self.day_off_day == 0 {
            self.phase = Phase::DayOffOffer;
            return;
        }
        self.advance();
    }

    /// `m4()`: 스왑 5, 빈칸 보충(열마다 위에서 첫 막힌 칸 위쪽을 아래부터 채움)
    fn morning_refill(&mut self) {
        self.swaps = 5;
        if self.closed_day() {
            return;
        }
        for x in 1..=COLS as i32 {
            let mut i = 1;
            while i <= ROWS as i32 + 1 {
                if !self.is_ground(x, i) || self.tile_at(x, i).is_some() || self.is_slot(x, i) {
                    i -= 1;
                    while i > 0 {
                        let id = self.new_tile(Kind::Wood);
                        let k = self.random_resource();
                        let t = &mut self.tiles[id];
                        t.kind = k;
                        t.gx = x;
                        t.gy = i;
                        self.set_grid(x, i, Some(id));
                        i -= 1;
                    }
                    break;
                }
                i += 1;
            }
        }
    }

    /// 휴식일 제안에 답한다
    pub fn answer_day_off(&mut self, accept: bool) {
        assert_eq!(self.phase, Phase::DayOffOffer);
        if accept {
            self.day_off_day = self.day;
        }
        self.phase = Phase::Fall;
        self.advance();
    }

    /// 휴식일 "Done": 스왑을 0으로 만들어 밤 직전(Dusk)으로 간다 (원본 `r7 = r6 = 0, m$()`)
    pub fn day_off_done(&mut self) -> ActionResult {
        if self.phase != Phase::Idle || !self.closed_day() {
            return ActionResult::Invalid;
        }
        self.swaps = 0;
        self.after_resolution();
        ActionResult::Ok
    }

    /// 점수 = day + 1000 × 업적
    pub fn score(&self) -> i64 {
        self.day as i64 + 1000 * self.achievements as i64
    }
}
