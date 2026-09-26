//! 타일 종류. 순서는 원본 목록 `ab`의 생성 순서와 같아야 한다(난수 추출이 이 순서를 쓴다).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Anvil = 0,       // a  os
    CannonSlot,      // C  on
    Tnt,             // T  o0
    Iceberg,         // I  og
    Fairy,           // F  e7
    FairyHouse,      // H  e5
    TimeMachine,     // M  ek
    You,             // Y  eb
    Wood,            // l  oh
    Ballista,        // b  oo
    Stone,           // s  om
    ArrowTower,      // t  ou
    Iron,            // i  od
    Cannon,          // c  or
    Treasure,        // g  o_
    Chest,           // h  ol
    IceCube,         // d  o$
    IceWall,         // w  of
}

pub const ALL_KINDS: [Kind; 18] = [
    Kind::Anvil,
    Kind::CannonSlot,
    Kind::Tnt,
    Kind::Iceberg,
    Kind::Fairy,
    Kind::FairyHouse,
    Kind::TimeMachine,
    Kind::You,
    Kind::Wood,
    Kind::Ballista,
    Kind::Stone,
    Kind::ArrowTower,
    Kind::Iron,
    Kind::Cannon,
    Kind::Treasure,
    Kind::Chest,
    Kind::IceCube,
    Kind::IceWall,
];

impl Kind {
    pub fn id(self) -> char {
        match self {
            Kind::Anvil => 'a',
            Kind::CannonSlot => 'C',
            Kind::Tnt => 'T',
            Kind::Iceberg => 'I',
            Kind::Fairy => 'F',
            Kind::FairyHouse => 'H',
            Kind::TimeMachine => 'M',
            Kind::You => 'Y',
            Kind::Wood => 'l',
            Kind::Ballista => 'b',
            Kind::Stone => 's',
            Kind::ArrowTower => 't',
            Kind::Iron => 'i',
            Kind::Cannon => 'c',
            Kind::Treasure => 'g',
            Kind::Chest => 'h',
            Kind::IceCube => 'd',
            Kind::IceWall => 'w',
        }
    }

    /// `upgradedVersion`
    pub fn upgraded(self) -> Option<Kind> {
        match self {
            Kind::Anvil => Some(Kind::Anvil),
            Kind::Iceberg => Some(Kind::Iceberg),
            Kind::Fairy => Some(Kind::FairyHouse),
            Kind::Wood | Kind::Ballista => Some(Kind::Ballista),
            Kind::Stone | Kind::ArrowTower => Some(Kind::ArrowTower),
            Kind::Iron | Kind::Cannon => Some(Kind::Cannon),
            Kind::Treasure | Kind::Chest => Some(Kind::Chest),
            Kind::IceCube | Kind::IceWall => Some(Kind::IceWall),
            _ => None,
        }
    }

    /// `resourceIs`: 상위 버전이 자기 자신이 아니고 Fairy도 아닌 것
    pub fn is_resource(self) -> bool {
        matches!(self, Kind::Wood | Kind::Stone | Kind::Iron | Kind::Treasure | Kind::IceCube)
    }

    pub fn unmovable(self) -> bool {
        self == Kind::CannonSlot
    }

    /// (피해, 재장전 프레임). 사격하는 타워만.
    pub fn weapon(self) -> Option<(f64, f64)> {
        match self {
            Kind::Ballista => Some((6.0, 196.0)),
            Kind::ArrowTower => Some((3.0, 55.0)),
            Kind::Cannon => Some((6.0, 98.0)),
            _ => None,
        }
    }
}
