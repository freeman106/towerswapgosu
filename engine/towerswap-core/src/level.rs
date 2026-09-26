//! 레벨(캐슬) 정의. 지금은 기본 레벨 First만 지원한다.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terrain {
    Water,
    Grass,
    Road,
    CannonSlot,
}

pub const COLS: usize = 6;
pub const ROWS: usize = 7; // 원본 rl

/// First 레벨: 1–6행 땅, 7행 물 (원본 기본 boardString)
pub fn first_level_terrain() -> [[Terrain; ROWS + 1]; COLS + 1] {
    let mut t = [[Terrain::Water; ROWS + 1]; COLS + 1];
    for x in 1..=COLS {
        for y in 1..=ROWS {
            t[x][y] = if y <= 6 { Terrain::Grass } else { Terrain::Water };
        }
    }
    t
}
