//! 보드 분석(패턴 검사): 교환 몇 번으로 상자 합성이 가능한지. 엔진을 돌리지 않는 근사다.
//! 이웃 두 타일의 교환만 본다(빈칸 이동, 모루, 요정으로 하는 합성과 중력·연쇄는 무시).
//! 지표(합성 기회와 선택 비율)와 연습 시작 상태 고르기에 쓴다.
use crate::game::Game;
use crate::kinds::Kind;
use crate::level::{Terrain, COLS, ROWS};

/// 칸 하나: (종류, 등급, 합성 가능 여부)
type Cell = Option<(Kind, u8, bool)>;

/// 패턴 검사용 보드 사본 (x 1..6, y 1..7)
#[derive(Clone)]
pub struct Board {
    c: [[Cell; ROWS + 2]; COLS + 2],
    slot: [[bool; ROWS + 2]; COLS + 2],
}

impl Board {
    pub fn of(g: &Game) -> Board {
        let mut b = Board { c: [[None; ROWS + 2]; COLS + 2], slot: [[false; ROWS + 2]; COLS + 2] };
        for x in 1..=COLS {
            for y in 1..=ROWS {
                b.slot[x][y] = g.terrain[x][y] == Terrain::CannonSlot;
                if let Some(t) = g.grid[x][y] {
                    let t = &g.tiles[t];
                    let mergeable = t.kind.upgraded().is_some() && t.tier < 4 && !b.slot[x][y];
                    b.c[x][y] = Some((t.kind, t.tier, mergeable));
                }
            }
        }
        b
    }

    /// 원본 gq(교환 유효성)의 패턴 버전
    fn swap_ok(&self, (ax, ay): (usize, usize), (bx, by): (usize, usize)) -> bool {
        let (Some(a), Some(b)) = (self.c[ax][ay], self.c[bx][by]) else { return false };
        if a.0 == Kind::CannonSlot || b.0 == Kind::CannonSlot {
            return false;
        }
        (a.0 == Kind::Iceberg) == (b.0 == Kind::Iceberg)
            && (a.0 != b.0 || a.1 != b.1)
            && (!(self.slot[ax][ay] || self.slot[bx][by]) || (a.0 == Kind::Cannon && b.0 == Kind::Cannon))
    }

    /// (x, y)를 지나는 같은 (종류, 등급) 줄이 3 이상이면 그 칸의 타일
    fn line_at(&self, x: usize, y: usize) -> Option<(Kind, u8)> {
        let (k, t, ok) = self.c[x][y]?;
        if !ok {
            return None;
        }
        let same = |x: usize, y: usize| matches!(self.c[x][y], Some((k2, t2, true)) if k2 == k && t2 == t);
        let run = |dx: i32, dy: i32| {
            let (mut n, mut px, mut py) = (0, x as i32 + dx, y as i32 + dy);
            while (1..=COLS as i32).contains(&px) && (1..=ROWS as i32).contains(&py) && same(px as usize, py as usize) {
                n += 1;
                px += dx;
                py += dy;
            }
            n
        };
        (1 + run(1, 0) + run(-1, 0) >= 3 || 1 + run(0, 1) + run(0, -1) >= 3).then_some((k, t))
    }

    /// 교환 한 번으로 합쳐지는 상자 등급 (비트 1 << 등급)
    pub fn one_swap_chest_merges(&self) -> u8 {
        let mut bits = 0u8;
        self.for_each_swap(|b, a, c| {
            for (x, y) in [a, c] {
                if let Some((Kind::Chest, t)) = b.line_at(x, y) {
                    bits |= 1 << t.min(4);
                }
            }
            false
        });
        bits
    }

    /// 상자 합성까지 필요한 교환 수 (1, 2) 또는 None (2보다 많음)
    pub fn chest_merge_distance(&self) -> Option<u32> {
        if self.one_swap_chest_merges() != 0 {
            return Some(1);
        }
        let mut found = false;
        self.for_each_swap(|b, _, _| {
            found = b.one_swap_chest_merges() != 0;
            found
        });
        found.then_some(2)
    }

    /// 등급별 합성 거리: (교환 한 번으로 합쳐지는 등급 비트, 두 번 이내로 합쳐지는 등급 비트)
    pub fn chest_merge_bits(&self) -> (u8, u8) {
        let d1 = self.one_swap_chest_merges();
        let mut d2 = d1;
        self.for_each_swap(|b, _, _| {
            d2 |= b.one_swap_chest_merges();
            false
        });
        (d1, d2)
    }

    /// 가능한 모든 이웃 교환을 적용한 보드로 f를 부른다. f가 참을 돌려주면 멈춘다.
    fn for_each_swap(&self, mut f: impl FnMut(&Board, (usize, usize), (usize, usize)) -> bool) {
        let mut b = self.clone();
        for x in 1..=COLS {
            for y in 1..=ROWS {
                for (dx, dy) in [(1, 0), (0, 1)] {
                    let (x2, y2) = (x + dx, y + dy);
                    if x2 > COLS || y2 > ROWS || !self.swap_ok((x, y), (x2, y2)) {
                        continue;
                    }
                    let (p, q) = (b.c[x][y], b.c[x2][y2]);
                    b.c[x][y] = q.map(|(k, t, _)| (k, t, k.upgraded().is_some() && t < 4 && !b.slot[x][y]));
                    b.c[x2][y2] = p.map(|(k, t, _)| (k, t, k.upgraded().is_some() && t < 4 && !b.slot[x2][y2]));
                    let stop = f(&b, (x, y), (x2, y2));
                    b.c[x][y] = self.c[x][y];
                    b.c[x2][y2] = self.c[x2][y2];
                    if stop {
                        return;
                    }
                }
            }
        }
    }

    /// 보드의 등급별 상자 수
    pub fn chest_counts(&self) -> [u32; 5] {
        let mut n = [0; 5];
        for col in &self.c {
            for &(k, t, _) in col.iter().flatten() {
                if k == Kind::Chest {
                    n[t.min(4) as usize] += 1;
                }
            }
        }
        n
    }
}
