//! 원본 `ul()` 난수기(xorshift32)를 비트 단위로 재현한다.
//! 원본 JS는 seed를 int32로 다루고 결과를 `>>> 0`(uint32)으로 반환한다.
//! 비트 패턴이 같으므로 u32로 저장한다.

#[derive(Clone, Debug)]
pub struct JsRng {
    pub seed: u32,
}

impl JsRng {
    pub fn new(seed: u32) -> Self {
        // 원본은 seed가 0이면 시각으로 다시 시드한다. 0은 쓰지 않는다.
        assert!(seed != 0, "seed 0은 허용하지 않음");
        JsRng { seed }
    }

    /// `_seedGo`
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut s = self.seed;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.seed = s;
        s
    }

    /// `floatGet`
    #[inline]
    pub fn float(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }

    /// `intGet(a, b)` 정수 인자 버전
    #[inline]
    pub fn int(&mut self, a: i64, b: i64) -> i64 {
        a + (self.next_u32() as i64) % (b - a + 1)
    }

    /// `intGet(a, b)` 실수 인자 버전 (원본은 b가 비정수인 호출이 있다). JS `%`와 f64 `%`는 같다.
    #[inline]
    pub fn int_f(&mut self, a: f64, b: f64) -> f64 {
        a + (self.next_u32() as f64) % (b - a + 1.0)
    }

    /// `itemGet(arr)` — 인덱스를 반환
    #[inline]
    pub fn item_index(&mut self, len: usize) -> usize {
        self.int(0, len as i64 - 1) as usize
    }
}
