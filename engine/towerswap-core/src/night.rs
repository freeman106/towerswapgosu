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
    pub gx2: i32,
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

#[derive(Clone, Default)]
pub struct NightState {
    pub dragons: Vec<Dragon>,
    pub a4: Vec<usize>, // 살아 있는(목록에 있는) 드래곤, 생성 순서
    pub projectiles: Vec<Projectile>,
    pub a6: Vec<usize>,
    pub ne: i64, // 원본 nE (상태 0,1,2,3,25에서 프레임마다 증가)
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
            gx2: 0,
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
                d.gx2 = gx;
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
                d.gx2 = gx;
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
                d.gx2 = rw;
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
        d.gx2 = d.gx;
    }

    /// `speedNowGet`
    fn dragon_speed_now(&self, id: usize) -> f64 {
        let d = &self.night.dragons[id];
        if d.is_dead {
            return 0.0;
        }
        let mut e = d.speed;
        if let Some(t) = self.tile_at(d.gx, d.gy) {
            let tile = &self.tiles[t];
            if tile.kind == Kind::IceWall || tile.kind == Kind::Iceberg {
                let i = if tile.kind == Kind::Iceberg { 0.5 } else { 0.1 };
                let f = js_mod(d.py / 48.0, 1.0);
                if f > i && f < i + 0.5 {
                    e *= 1.0 - (0.42 + 0.12 * tile.tier as f64);
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

    fn remove_dragon(&mut self, id: usize) {
        self.night.a4.retain(|&d| d != id);
    }

    /// `damageTheCastle`. 하트가 0이 되면 참(게임 오버)
    fn damage_castle(&mut self, id: usize) -> bool {
        self.hearts -= 1;
        if self.hearts <= 0 {
            self.phase = Phase::GameOver;
            return true;
        }
        self.remove_dragon(id);
        false
    }

    /// `mD.prototype.update2`. 게임 오버면 참
    fn dragon_update(&mut self, id: usize) -> bool {
        if self.night.dragons[id].is_dead {
            let d = &self.night.dragons[id];
            if d.got_to_castle {
                if self.loops - d.dead_time > 15 {
                    return self.damage_castle(id);
                }
            } else {
                let d = &mut self.night.dragons[id];
                d.py += 0.3 * d.speed;
            }
            if self.loops - self.night.dragons[id].dead_time > 20 {
                self.remove_dragon(id);
            }
            return false;
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
            return false;
        }
        if d.kind == DragonKind::Mother && d.py < 0.0 {
            self.dragon_die(id);
        }
        false
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
            Kind::Ballista => tile.gx >= d.gx && tile.gx <= d.gx2,
            Kind::Cannon => tile.gy == d.gy,
            Kind::ArrowTower => {
                let r = if self.is_turret(t) { 2 } else { 1 };
                tile.gx - r <= d.gx2 && tile.gx + r >= d.gx && r >= (tile.gy - d.gy).abs()
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

    /// `ua.prototype.update6` (밤)
    fn tower_update(&mut self, t: TileId) {
        if self.tiles[t].reload <= 0.0 && self.tiles[t].kind.weapon().is_some() {
            let mut best: Option<(usize, f64)> = None;
            for idx in 0..self.night.a4.len() {
                let e = self.night.a4[idx];
                if self.can_shoot(t, e) {
                    let v = self.shoot_value(t, e);
                    match best {
                        None => best = Some((e, v)),
                        Some((_, bv)) if v > bv => best = Some((e, v)),
                        _ => {}
                    }
                }
            }
            if let Some((e, _)) = best {
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

    fn remove_projectile(&mut self, p: usize) {
        self.night.a6.retain(|&x| x != p);
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
            self.remove_projectile(p);
            true
        }
    }

    /// `mL.prototype.update3`
    fn proj_update(&mut self, p: usize) {
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
            if self.night.projectiles[p].exploded > 15 {
                self.remove_projectile(p);
            }
            return;
        }
        let kind = self.night.projectiles[p].tower_kind;
        if kind != Kind::Cannon {
            if let Some(t) = self.night.projectiles[p].target {
                if self.night.dragons[t].is_dead {
                    let deleted = self.proj_retarget(p);
                    if kind != Kind::Ballista || deleted {
                        return;
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
                    self.remove_projectile(p);
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
        loop {
            self.night.ne += 1;
            // 1) 포탑, 이어서 타일 목록 순서대로 사격
            for x in 1..=COLS {
                if let Some(t) = self.turrets[x] {
                    self.tower_update(t);
                }
            }
            for i in 0..self.ak.len() {
                let t = self.ak[i];
                self.tower_update(t);
            }
            // 2) 드래곤 이동 (순회 중 현재 항목 삭제, 끝에 추가 가능)
            let mut i = 0;
            while i < self.night.a4.len() {
                let d = self.night.a4[i];
                if self.dragon_update(d) {
                    return; // 게임 오버
                }
                if self.night.a4.get(i) == Some(&d) {
                    i += 1;
                }
            }
            // 3) 투사체
            let mut i = 0;
            while i < self.night.a6.len() {
                let p = self.night.a6[i];
                self.proj_update(p);
                if self.night.a6.get(i) == Some(&p) {
                    i += 1;
                }
            }
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
