//! 파이썬 바인딩: N개 게임을 rayon 스레드로 병렬 진행하는 배치 환경.
//! 관측·마스크·보상은 파이썬이 미리 할당한 numpy 배열에 바로 쓴다(복사 없음).
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use towerswap_core::analysis::Board;
use towerswap_core::env::{Env, OpenRule, GRID_H, GRID_LEN, GRID_W, N_ACTIONS, N_GRID_CH, N_SCALAR};
use towerswap_core::expert::{greedy_set, greedy_set_with, invest_not_worse, params, path_not_worse, Params};
use towerswap_core::kinds::Kind;
use towerswap_core::rng::JsRng;
use towerswap_core::{Dir, Game, Phase};

/// 드래그 행동 번호의 끝 (0..168)
const A_DRAG_END: usize = 168;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// 게임 하나와 그 게임의 시드 흐름
struct Slot {
    env: Env,
    rng: u64,
    steps: u64,
    erng: u64, // 전문가 동점 깨기
    chest_bonus: [f32; 5],
    econ: (f32, f32),         // 경제 조형 (가중치, 하루 할인)
    no_toss_day_off: bool,    // 휴식일 버리기 금지
    open_rule: OpenRule,      // 상자 개봉 규칙
    merge_rule: Option<(i64, u32)>, // 합성 우선 규칙 (남은 스왑 하한, 하루 최대 횟수)
    practice: u8,            // 지금 게임의 연습 시작 난이도 (0: 정상 시작)
    aux: Option<Vec<bool>>,   // 지금 상태가 보조 손실 대상이면 그 행동 집합 M(s)
    aux_prep: bool,           // aux가 준비 이동 집합이면 참: 그중 하나를 고르면 다음 상태에 합성 행동 집합을 붙인다
    aux_hold: Option<i32>,    // 보관 연습(5단계) 시작일: 확인 구간(이틀) 동안 매 결정에 투자 greedy의 최선 행동 집합을 붙인다
}

/// 연습 시작 상태. `merge`는 합성이 유리하다고 확인된 경우의 합성 행동 집합 M(s)
struct Start {
    game: Game,
    level: u8,
    merge: Option<Vec<bool>>, // 1단계: 합성 행동 집합, 2단계: 준비 이동 집합, 5단계: 투자 greedy의 최선 행동 집합 (확인된 경우만)
}

impl Slot {
    /// 새 게임. 확률 `frac`로 연습 시작 풀의 상태에서 시작한다(난수는 새로 뽑고 통계는 비운다).
    fn new_game(&mut self, starts: &[Start], frac: f64) {
        let v = splitmix(&mut self.rng) as u32 | 1; // 시드 0은 허용하지 않는다
        let m = splitmix(&mut self.rng) as u32 | 1;
        let r = splitmix(&mut self.rng);
        let practice = !starts.is_empty() && ((r >> 11) as f64 / (1u64 << 53) as f64) < frac;
        self.aux = None;
        self.aux_prep = false;
        self.aux_hold = None;
        self.practice = 0;
        self.env = if practice {
            let st = &starts[(splitmix(&mut self.rng) % starts.len() as u64) as usize];
            self.aux = st.merge.clone();
            self.aux_prep = st.merge.is_some() && st.level == 2;
            self.aux_hold = (st.merge.is_some() && st.level == 5).then_some(st.game.day);
            self.practice = st.level;
            let mut g = st.game.clone();
            g.rng_v = JsRng::new(v);
            g.rng_m = JsRng::new(m);
            g.reset_stats();
            Env::from_game(g)
        } else {
            Env::new(v, m)
        };
        self.env.chest_bonus = self.chest_bonus;
        (self.env.econ_w, self.env.econ_gamma) = self.econ;
        self.env.no_toss_day_off = self.no_toss_day_off;
        self.env.open_rule = self.open_rule;
        self.env.merge_rule = self.merge_rule;
        self.steps = 0;
    }
}

/// 연습 시작 상태의 조건 (입력 대기, 스왑이 남은 평일)
///   1: 교환 한 번이면 상자 합성, 2: 교환 두 번, 3: 같은 등급 상자 3개 이상(교환 세 번 이상), 4: 같은 등급 상자 2개
fn start_level(g: &Game) -> Option<u32> {
    if g.phase != Phase::Idle || g.swaps < 1 || g.day == g.day_off_day {
        return None;
    }
    let b = Board::of(g);
    let n = b.chest_counts();
    let most = n[1..4].iter().copied().max().unwrap_or(0);
    if most >= 3 {
        return Some(b.chest_merge_distance().unwrap_or(3));
    }
    (most == 2).then_some(4)
}

/// 상태에서 상자 합성을 일으키는 유효 행동 (엔진으로 시험)
fn merge_set(env: &Env) -> Vec<bool> {
    let mut m = [false; N_ACTIONS];
    env.mask(&mut m);
    let made0: u32 = env.game.chest_made[2..].iter().sum();
    (0..N_ACTIONS)
        .map(|a| {
            m[a] && {
                let mut e = env.clone();
                e.step(a).unwrap();
                e.game.chest_made[2..].iter().sum::<u32>() > made0
            }
        })
        .collect()
}

/// 다음 수에 상자 합성이 가능해지는 준비 이동 집합과 경로 예시 (준비 이동, 합성 행동). 엔진으로 확인한다
fn prep_set(env: &Env) -> (Vec<bool>, Option<(usize, usize)>) {
    let mut m = [false; N_ACTIONS];
    env.mask(&mut m);
    let mut example = None;
    let set = (0..N_ACTIONS)
        .map(|a| {
            if !m[a] {
                return false;
            }
            let mut e = env.clone();
            e.step(a).unwrap();
            if e.done() || e.game.day != env.game.day || e.game.phase != Phase::Idle || !e.chest_merge_available() {
                return false;
            }
            let ms = merge_set(&e);
            match ms.iter().position(|&x| x) {
                Some(b) => {
                    example.get_or_insert((a, b));
                    true
                }
                None => false,
            }
        })
        .collect();
    (set, example)
}

/// 합성 대신 상자를 여는 행동: 합성 경로 `path`로 합쳐지는 등급의 상자 탭 중 첫 번째
fn open_instead(env: &Env, path: &[usize]) -> Option<usize> {
    let g = &env.game;
    let mut m = [false; N_ACTIONS];
    env.mask(&mut m);
    let mut e = env.clone();
    for &a in path {
        e.step(a)?;
    }
    let tier = (2..=4).find(|&t| e.game.chest_made[t] > g.chest_made[t])? as u8 - 1;
    (towerswap_core::env::A_CELL..towerswap_core::env::A_YES).find(|&a| {
        let c = a - towerswap_core::env::A_CELL;
        m[a] && g.grid[c % 6 + 1][c / 6].map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier == tier)
    })
}

/// 보관 연습(5단계) 조건: 입력 대기, 스왑 2 이상인 평일, 일반 상자 1~2개, 교환 한 번 합성은 없음
fn hold_start(g: &Game) -> bool {
    if g.phase != Phase::Idle || g.swaps < 2 || g.day == g.day_off_day {
        return false;
    }
    let b = Board::of(g);
    (1..=2).contains(&b.chest_counts()[1]) && b.one_swap_chest_merges() == 0
}

/// 보관 연습의 대상: 투자 greedy의 최선 행동 집합 (입력 대기 상태에서)
fn hold_targets(env: &Env, rng: &mut u64) -> Vec<bool> {
    let (mut m, mut set) = ([false; N_ACTIONS], vec![false; N_ACTIONS]);
    env.mask(&mut m);
    greedy_set_with(env, &m, &mut set, rng, &Params::invest());
    set
}

/// 일반 상자를 여는 행동 중 첫 번째
fn open_normal(env: &Env) -> Option<usize> {
    let g = &env.game;
    let mut m = [false; N_ACTIONS];
    env.mask(&mut m);
    (towerswap_core::env::A_CELL..towerswap_core::env::A_YES).find(|&a| {
        let c = a - towerswap_core::env::A_CELL;
        m[a] && g.grid[c % 6 + 1][c / 6].map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier == 1)
    })
}

/// 게임을 두어 조건 `level`에 맞는 상태를 하나 찾는다. 1~4단계는 상자를 모으는 greedy,
/// 5단계(보관 연습)는 기본 greedy(지금 정책처럼 상자를 바로 여는 플레이)의 게임에서 찾는다.
fn find_start(level: u32, seed: u64) -> Game {
    let (mut rng, p) = (seed, if level == 5 { params().clone() } else { Params::hoarding() });
    let (mut m, mut set) = ([false; N_ACTIONS], [false; N_ACTIONS]);
    loop {
        let mut env = Env::new(splitmix(&mut rng) as u32 | 1, splitmix(&mut rng) as u32 | 1);
        // 5단계는 목표일(1~20일) 이후의 첫 상태를 받아 날짜가 고루 퍼지게 한다
        let from_day = if level == 5 { 1 + (splitmix(&mut rng) % 20) as i32 } else { 0 };
        while !env.done() && env.game.day < 60 {
            // 맞는 상태를 절반 확률로 받는다(게임 안의 여러 시점이 고루 뽑히도록)
            let ok = if level == 5 { env.game.day >= from_day && hold_start(&env.game) } else { start_level(&env.game) == Some(level) };
            if ok && (level == 5 || splitmix(&mut rng) % 2 == 0) {
                return env.game.clone();
            }
            env.mask(&mut m);
            let a = greedy_set_with(&env, &m, &mut set, &mut rng, &p);
            env.step(a).unwrap();
        }
    }
}

/// 끝난 게임 기록
struct Finished {
    score: i64,
    day: i32,
    steps: u64,
    truncated: bool,
    env: usize,
    max_chest: u8,
    r_survival: f32,
    r_boss: f32,
    r_chest: f32,
    r_econ: f32,
    made: [u32; 4],   // 등급 1..4 상자 생성 수
    opened: [u32; 4], // 등급 1..4 상자 개봉 수
    hold: [(u64, u64); 4], // 등급 1..4 개봉한 상자의 보관 기간 합 (행동 수, 날)
    merge_opps: u32,
    merge_taken: u32,
    max_normal_held: u32,
    practice: u8,
    first_merge: Option<(u32, i32)>,
    steps_hold2: u32,
    steps_hold3: u32,
    max_same_held: u32,
    dmg: [[f64; 5]; 4],
    tower_nights: [[u32; 5]; 4],
    dmg_row: [[f64; 8]; 4],
    tn_row: [[u32; 8]; 4],
    made_tw: [[u32; 5]; 5],
    made_tw_row: [[u32; 8]; 5],
    emergency_opens: u32,
    start_day: i32,
    merge_rule_uses: u32,
}

impl Finished {
    fn of(slot: &Slot, env: usize, truncated: bool) -> Finished {
        let (e, g) = (&slot.env, &slot.env.game);
        Finished {
            score: g.score(),
            day: g.day,
            steps: slot.steps,
            truncated,
            env,
            max_chest: e.max_chest_tier(),
            r_survival: e.ep_survival,
            r_boss: e.ep_boss,
            r_chest: e.ep_chest,
            r_econ: e.ep_econ,
            made: g.chest_made[1..].try_into().unwrap(),
            opened: g.chest_opened[1..].try_into().unwrap(),
            hold: g.chest_hold[1..].try_into().unwrap(),
            merge_opps: e.merge_opps,
            merge_taken: e.merge_taken,
            max_normal_held: e.max_normal_held,
            practice: slot.practice,
            first_merge: e.first_merge,
            steps_hold2: e.steps_hold2,
            steps_hold3: e.steps_hold3,
            max_same_held: e.max_same_held,
            dmg: g.stat_dmg,
            tower_nights: g.stat_tower_nights,
            dmg_row: g.stat_dmg_row,
            tn_row: g.stat_tn_row,
            made_tw: g.stat_made,
            made_tw_row: g.stat_made_row,
            emergency_opens: e.emergency_opens,
            start_day: e.start_day(),
            merge_rule_uses: e.merge_rule_uses,
        }
    }
}

fn slice_mut<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a mut numpy::PyReadwriteArray<'_, T, D>,
    len: usize,
    name: &str,
) -> PyResult<&'a mut [T]> {
    let s = a.as_slice_mut().map_err(|e| PyValueError::new_err(format!("{name}: {e}")))?;
    if s.len() != len {
        return Err(PyValueError::new_err(format!("{name}: 길이 {} (기대 {len})", s.len())));
    }
    Ok(s)
}

#[pyclass]
struct VecEnv {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    max_steps: u64,
    finished: Vec<Finished>,
    starts: Vec<Start>, // 연습 시작 상태
    start_frac: f64,
}

#[pymethods]
impl VecEnv {
    /// n: 게임 수, seed: 시드, threads: 0이면 CPU 수, max_steps: 한 게임의 스텝 상한(안전장치),
    /// chest_bonus: 등급 1..4 상자를 게임에서 처음 만들 때의 보상 [b1, b2, b3, b4],
    /// econ_w, econ_gamma: 경제 조형 가중치(스왑 1개당, 0이면 끔)와 하루 할인(학습의 γ와 같게),
    /// no_toss_day_off: 휴식일 버리기 금지 (행동 마스크 제약),
    /// 상자 개봉 규칙 (행동 마스크 제약): open_min_tier 미만 등급은 열 수 없다(no_open_normal=True는 open_min_tier=2와 같다).
    /// emergency=(등급, 남은 스왑 상한, 하트 상한)이면 그 등급 상자는 스왑·하트가 상한 이하이고 지금 열 수 있는
    /// open_min_tier 이상 상자가 없을 때만 열 수 있다.
    /// merge_rule=(남은 스왑 하한, 하루 최대 횟수): 은상자 이상을 연 날 기본→동 무기 합성 드래그가 있으면 행동을 그 드래그로 제한한다
    #[new]
    #[pyo3(signature = (n, seed = 1, threads = 0, max_steps = 200_000, chest_bonus = None, econ_w = 0.0, econ_gamma = 1.0, no_toss_day_off = false, no_open_normal = false, open_min_tier = 1, emergency = None, merge_rule = None))]
    fn new(
        n: usize,
        seed: u64,
        threads: usize,
        max_steps: u64,
        chest_bonus: Option<Vec<f32>>,
        econ_w: f32,
        econ_gamma: f32,
        no_toss_day_off: bool,
        no_open_normal: bool,
        open_min_tier: u8,
        emergency: Option<(u8, i64, i32)>,
        merge_rule: Option<(i64, u32)>,
    ) -> PyResult<Self> {
        let open_rule = OpenRule { min_tier: open_min_tier.max(if no_open_normal { 2 } else { 1 }), emergency };
        let mut cb = [0.0f32; 5];
        if let Some(b) = chest_bonus {
            if b.len() != 4 {
                return Err(PyValueError::new_err("chest_bonus: 등급 1..4의 값 4개"));
            }
            cb[1..].copy_from_slice(&b);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut root = seed;
        let slots = (0..n)
            .map(|_| {
                let mut s = Slot {
                    env: Env::new(1, 1),
                    rng: splitmix(&mut root),
                    steps: 0,
                    erng: splitmix(&mut root),
                    chest_bonus: cb,
                    econ: (econ_w, econ_gamma),
                    no_toss_day_off,
                    open_rule,
                    merge_rule,
                    practice: 0,
                    aux: None,
                    aux_prep: false,
                    aux_hold: None,
                };
                s.new_game(&[], 0.0);
                s
            })
            .collect();
        Ok(VecEnv { slots, pool, max_steps, finished: Vec::new(), starts: Vec::new(), start_frac: 0.0 })
    }

    /// 연습 시작 상태 `count`개를 만든다(실제 게임에서). 기존 풀은 바꾼다.
    /// level 1: 교환 한 번이면 상자 합성, 2: 교환 두 번, 3: 같은 등급 상자 3개 이상, 4: 같은 등급 상자 2개,
    /// 5: 보관 연습(기본 greedy 게임에서 일반 상자 1~2개를 가진 평일 낮. 확인은 투자 대 지금 개봉을 밤 2번까지 비교)
    /// confirm_samples > 0이면(난이도 1·2) 상태마다 합성 경로(1단계: 합성, 2단계: 준비 이동 + 합성)와 상자 개봉을
    /// 그날 밤까지 굴려 비교하고(시드 쌍 수), 합성 쪽이 나쁘지 않은 상태에만 보조 손실 대상 M(s)을 붙인다
    /// (1단계: 합성 행동 집합, 2단계: 준비 이동 집합). append면 기존 풀에 더한다.
    #[pyo3(signature = (count, level, seed = 1, confirm_samples = 0, append = false))]
    fn build_start_pool(&mut self, py: Python<'_>, count: usize, level: u32, seed: u64, confirm_samples: u32, append: bool) -> PyResult<()> {
        if !(1..=5).contains(&level) {
            return Err(PyValueError::new_err("level: 1..5"));
        }
        let pool = &self.pool;
        let new: Vec<Start> = py.detach(|| {
            pool.install(|| {
                (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let mut rng = splitmix(&mut (seed ^ (i as u64) << 20));
                        let game = find_start(level, rng);
                        let mut merge = None;
                        if level == 5 && confirm_samples > 0 {
                            let env = Env::from_game(game.clone());
                            if let Some(open) = open_normal(&env) {
                                if invest_not_worse(&env, open, 2, confirm_samples, 1.0, &mut rng) {
                                    merge = Some(hold_targets(&env, &mut rng));
                                }
                            }
                        }
                        if level <= 2 && confirm_samples > 0 {
                            let env = Env::from_game(game.clone());
                            let (set, path) = if level == 1 {
                                let m = merge_set(&env);
                                let p = m.iter().position(|&x| x).map(|a| vec![a]);
                                (m, p)
                            } else {
                                let (m, ex) = prep_set(&env);
                                (m, ex.map(|(a, b)| vec![a, b]))
                            };
                            if let Some(path) = path {
                                if let Some(open) = open_instead(&env, &path) {
                                    if path_not_worse(&env, &path, open, confirm_samples, &mut rng) {
                                        merge = Some(set);
                                    }
                                }
                            }
                        }
                        Start { game, level: level as u8, merge }
                    })
                    .collect()
            })
        });
        if !append {
            self.starts.clear();
        }
        self.starts.extend(new);
        Ok(())
    }

    /// 현재 상태의 보조 손실 대상: flag bool [N](대상 여부), target bool [N, A](합성 행동 집합 M(s))
    fn aux_targets(&self, mut flag: PyReadwriteArray1<'_, bool>, mut target: PyReadwriteArrayDyn<'_, bool>) -> PyResult<()> {
        let n = self.slots.len();
        let f = slice_mut(&mut flag, n, "flag")?;
        let t = slice_mut(&mut target, n * N_ACTIONS, "target")?;
        for (i, slot) in self.slots.iter().enumerate() {
            f[i] = slot.aux.is_some();
            let row = &mut t[i * N_ACTIONS..(i + 1) * N_ACTIONS];
            match &slot.aux {
                Some(m) => row.copy_from_slice(m),
                None => row.fill(false),
            }
        }
        Ok(())
    }

    /// 새 게임을 연습 시작 상태에서 시작할 확률
    fn set_start_frac(&mut self, frac: f64) {
        self.start_frac = frac;
    }

    /// 연습 시작 상태들의 요약: (개수, day 평균, 보조 손실 대상 비율)
    fn start_pool_info(&self) -> (usize, f64, f64) {
        let n = self.starts.len().max(1) as f64;
        (
            self.starts.len(),
            self.starts.iter().map(|s| s.game.day as f64).sum::<f64>() / n,
            self.starts.iter().filter(|s| s.merge.is_some()).count() as f64 / n,
        )
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    /// 모든 게임을 새로 시작하고 관측과 마스크를 쓴다.
    /// grid: float32 [N, C, H, W], scal: float32 [N, S], mask: bool [N, A]
    fn reset(
        &mut self,
        py: Python<'_>,
        mut grid: PyReadwriteArrayDyn<'_, f32>,
        mut scal: PyReadwriteArrayDyn<'_, f32>,
        mut mask: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<()> {
        let n = self.slots.len();
        let g = slice_mut(&mut grid, n * GRID_LEN, "grid")?;
        let s = slice_mut(&mut scal, n * N_SCALAR, "scal")?;
        let m = slice_mut(&mut mask, n * N_ACTIONS, "mask")?;
        let (slots, pool, starts, frac) = (&mut self.slots, &self.pool, &self.starts, self.start_frac);
        py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .zip(g.par_chunks_mut(GRID_LEN))
                    .zip(s.par_chunks_mut(N_SCALAR))
                    .zip(m.par_chunks_mut(N_ACTIONS))
                    .for_each(|(((slot, g), s), m)| {
                        slot.new_game(starts, frac);
                        slot.env.write_obs(g, s);
                        slot.env.mask(m);
                    })
            })
        });
        Ok(())
    }

    /// 행동을 적용한다. 끝난 게임은 새 게임으로 바뀌고, 쓰는 관측은 새 게임의 첫 상태다.
    /// actions: int64 [N]. reward: float32 [N], done: bool [N], days: float32 [N](이 스텝에 지난 날 수)
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<'_, i64>,
        mut grid: PyReadwriteArrayDyn<'_, f32>,
        mut scal: PyReadwriteArrayDyn<'_, f32>,
        mut mask: PyReadwriteArrayDyn<'_, bool>,
        mut reward: PyReadwriteArray1<'_, f32>,
        mut done: PyReadwriteArray1<'_, bool>,
        mut days: PyReadwriteArray1<'_, f32>,
    ) -> PyResult<()> {
        let n = self.slots.len();
        let acts = actions.as_slice().map_err(|e| PyValueError::new_err(format!("actions: {e}")))?;
        if acts.len() != n {
            return Err(PyValueError::new_err(format!("actions: 길이 {} (기대 {n})", acts.len())));
        }
        let g = slice_mut(&mut grid, n * GRID_LEN, "grid")?;
        let s = slice_mut(&mut scal, n * N_SCALAR, "scal")?;
        let m = slice_mut(&mut mask, n * N_ACTIONS, "mask")?;
        let r = slice_mut(&mut reward, n, "reward")?;
        let d = slice_mut(&mut done, n, "done")?;
        let dy = slice_mut(&mut days, n, "days")?;
        let (slots, pool, max_steps, starts, frac) = (&mut self.slots, &self.pool, self.max_steps, &self.starts, self.start_frac);
        let out: Result<Vec<Option<Finished>>, String> = py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .enumerate()
                    .zip(g.par_chunks_mut(GRID_LEN))
                    .zip(s.par_chunks_mut(N_SCALAR))
                    .zip(m.par_chunks_mut(N_ACTIONS))
                    .zip(r.par_iter_mut())
                    .zip(d.par_iter_mut())
                    .zip(dy.par_iter_mut())
                    .zip(acts.par_iter())
                    .map(|((((((((i, slot), g), s), m), r), d), dy), &a)| {
                        let phase = slot.env.game.phase;
                        // 준비 이동 집합 중 하나를 골랐으면 다음 상태의 합성 행동 집합을 대상으로 붙인다
                        let prep_hit = slot.aux_prep && slot.aux.as_ref().map_or(false, |m| (a as usize) < N_ACTIONS && m[a as usize]);
                        slot.aux = None;
                        slot.aux_prep = false;
                        let o = slot.env.step(a as usize).ok_or_else(|| format!("환경 {i}: 무효 행동 {a} (단계 {phase:?})"))?;
                        slot.steps += 1;
                        *r = o.reward;
                        *dy = o.days as f32;
                        let truncated = !o.done && slot.steps >= max_steps;
                        let fin = (o.done || truncated).then(|| {
                            let f = Finished::of(slot, i, truncated);
                            slot.new_game(starts, frac);
                            f
                        });
                        *d = fin.is_some();
                        // 보관 연습: 확인 구간(시작일과 다음 날), 입력 대기, 1~3등급 상자가 있는 동안 계속 대상을 붙인다
                        if let Some(day) = slot.aux_hold {
                            let g = &slot.env.game;
                            let n = Board::of(g).chest_counts();
                            let keep = fin.is_none() && g.day < day + 2 && g.day != g.day_off_day && g.phase == Phase::Idle && n[1..4].iter().any(|&c| c > 0);
                            if keep {
                                slot.aux = Some(hold_targets(&slot.env, &mut slot.erng));
                            } else {
                                slot.aux_hold = None;
                            }
                        }
                        if prep_hit && fin.is_none() && slot.env.game.phase == Phase::Idle {
                            let ms = merge_set(&slot.env);
                            if ms.iter().any(|&x| x) {
                                slot.aux = Some(ms);
                            }
                        }
                        slot.env.write_obs(g, s);
                        slot.env.mask(m);
                        Ok(fin)
                    })
                    .collect()
            })
        });
        self.finished.extend(out.map_err(PyValueError::new_err)?.into_iter().flatten());
        Ok(())
    }

    /// 각 게임의 현재 상태에서 greedy 전문가를 계산한다.
    /// out: int64 [N] 최선 행동 집합에서 무작위로 고른 행동, best: bool [N, A] 최선 행동 집합(동점 전부)
    fn expert(&mut self, py: Python<'_>, mut out: PyReadwriteArray1<'_, i64>, mut best: PyReadwriteArrayDyn<'_, bool>) -> PyResult<()> {
        let n = self.slots.len();
        let o = slice_mut(&mut out, n, "out")?;
        let b = slice_mut(&mut best, n * N_ACTIONS, "best")?;
        let (slots, pool) = (&mut self.slots, &self.pool);
        py.detach(|| {
            pool.install(|| {
                slots.par_iter_mut().zip(o.par_iter_mut()).zip(b.par_chunks_mut(N_ACTIONS)).for_each(|((slot, o), b)| {
                    let mut m = [false; N_ACTIONS];
                    slot.env.mask(&mut m);
                    *o = greedy_set(&slot.env, &m, b, &mut slot.erng) as i64;
                })
            })
        });
        Ok(())
    }

    /// 진단: 각 게임의 현재 상태에서 상자 합성을 일으키는 유효 행동을 엔진으로 시험해 표시한다. out: bool [N, A]
    fn merge_actions(&self, py: Python<'_>, mut out: PyReadwriteArrayDyn<'_, bool>) -> PyResult<()> {
        let o = slice_mut(&mut out, self.slots.len() * N_ACTIONS, "out")?;
        let (slots, pool) = (&self.slots, &self.pool);
        py.detach(|| {
            pool.install(|| {
                slots.par_iter().zip(o.par_chunks_mut(N_ACTIONS)).for_each(|(slot, o)| {
                    o.fill(false);
                    let mut m = [false; N_ACTIONS];
                    slot.env.mask(&mut m);
                    let made0: u32 = slot.env.game.chest_made[2..].iter().sum();
                    for a in (0..N_ACTIONS).filter(|&a| m[a]) {
                        let mut e = slot.env.clone();
                        e.step(a).unwrap();
                        o[a] = e.game.chest_made[2..].iter().sum::<u32>() > made0;
                    }
                })
            })
        });
        Ok(())
    }

    /// 지난 호출 이후 끝난 게임 기록을 꺼낸다. 게임마다 dict:
    /// score, day, steps, truncated, env, max_chest, r_survival, r_boss, r_chest, r_econ(보상 성분 합계),
    /// made, opened(등급 1..4 상자 생성·개봉 수), hold_steps, hold_days(등급 1..4 개봉한 상자의 보관 기간 합),
    /// merge_opps, merge_taken(교환 한 번 상자 합성 기회와 실제 합성), max_normal_held,
    /// practice(연습 시작 여부), practice_level(연습 시작 난이도, 정상 시작은 0),
    /// first_merge_step, first_merge_day(첫 상자 합성까지의 행동 수와 지난 날, 없으면 -1),
    /// steps_hold2, steps_hold3(같은 등급 상자를 2개·3개 이상 가진 행동 수), max_same_held
    fn pop_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        std::mem::take(&mut self.finished)
            .into_iter()
            .map(|f| {
                let d = PyDict::new(py);
                d.set_item("score", f.score)?;
                d.set_item("day", f.day)?;
                d.set_item("steps", f.steps)?;
                d.set_item("truncated", f.truncated)?;
                d.set_item("env", f.env)?;
                d.set_item("max_chest", f.max_chest)?;
                d.set_item("r_survival", f.r_survival)?;
                d.set_item("r_boss", f.r_boss)?;
                d.set_item("r_chest", f.r_chest)?;
                d.set_item("r_econ", f.r_econ)?;
                d.set_item("made", f.made.to_vec())?;
                d.set_item("opened", f.opened.to_vec())?;
                d.set_item("hold_steps", f.hold.iter().map(|h| h.0).collect::<Vec<_>>())?;
                d.set_item("hold_days", f.hold.iter().map(|h| h.1).collect::<Vec<_>>())?;
                d.set_item("merge_opps", f.merge_opps)?;
                d.set_item("merge_taken", f.merge_taken)?;
                d.set_item("max_normal_held", f.max_normal_held)?;
                d.set_item("practice", f.practice > 0)?;
                d.set_item("practice_level", f.practice)?;
                d.set_item("first_merge_step", f.first_merge.map_or(-1, |x| x.0 as i64))?;
                d.set_item("first_merge_day", f.first_merge.map_or(-1, |x| x.1 as i64))?;
                d.set_item("steps_hold2", f.steps_hold2)?;
                d.set_item("steps_hold3", f.steps_hold3)?;
                d.set_item("max_same_held", f.max_same_held)?;
                d.set_item("dmg", f.dmg.iter().map(|r| r[1..].to_vec()).collect::<Vec<_>>())?;
                d.set_item("tower_nights", f.tower_nights.iter().map(|r| r[1..].to_vec()).collect::<Vec<_>>())?;
                d.set_item("dmg_row", f.dmg_row.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
                d.set_item("tn_row", f.tn_row.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
                d.set_item("made_tiers", f.made_tw.iter().map(|r| r[1..].to_vec()).collect::<Vec<_>>())?;
                d.set_item("made_rows", f.made_tw_row.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
                d.set_item("emergency_opens", f.emergency_opens)?;
                d.set_item("start_day", f.start_day)?;
                d.set_item("merge_rule_uses", f.merge_rule_uses)?;
                Ok(d)
            })
            .collect()
    }

    /// 탐색 환경 슬롯 src의 게임들을 연습 시작 상태(난이도 level)로 풀에 더한다. place_tier > 0이면 보드의 1~6행 자원 타일 하나를
    /// 그 등급 상자로 바꾼다(seed로 고르고, 자원이 없는 상태는 건너뛴다). 더한 수를 돌려준다. 풀을 쓰려면 set_start_frac으로 비율을 정한다
    #[pyo3(signature = (search, src, level, place_tier = 0, seed = 1))]
    fn add_starts(&mut self, search: PyRef<'_, SearchEnv>, src: PyReadonlyArray1<'_, i64>, level: u8, place_tier: u8, seed: u64) -> PyResult<usize> {
        let mut rng = seed;
        let mut added = 0;
        for &j in src.as_slice()? {
            let slot = search.slots.get(j as usize).ok_or_else(|| PyValueError::new_err(format!("add_starts: 슬롯 {j} 범위 밖")))?;
            let mut g = slot.env.game.clone();
            if place_tier > 0 && g.replace_resource_with_chest(place_tier, splitmix(&mut rng)).is_none() {
                continue;
            }
            self.starts.push(Start { game: g, level, merge: None });
            added += 1;
        }
        Ok(added)
    }

    /// 탐색 환경 슬롯 src[j]의 게임을 게임 dst[j]로 가져온다(되돌린 상태에서 이어 두기용). 연습 시작·보조 대상은 지운다.
    fn load_from(&mut self, search: PyRef<'_, SearchEnv>, src: PyReadonlyArray1<'_, i64>, dst: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let (src, dst) = (src.as_slice()?, dst.as_slice()?);
        if src.len() != dst.len() {
            return Err(PyValueError::new_err("load_from: src, dst 길이가 다르다"));
        }
        for (&j, &i) in src.iter().zip(dst) {
            let (j, i) = (j as usize, i as usize);
            if j >= search.slots.len() || i >= self.slots.len() {
                return Err(PyValueError::new_err(format!("load_from: 슬롯 {j} 또는 게임 {i} 범위 밖")));
            }
            let slot = &mut self.slots[i];
            slot.env = search.slots[j].env.clone();
            slot.env.chest_bonus = slot.chest_bonus;
            (slot.env.econ_w, slot.env.econ_gamma) = slot.econ;
            slot.env.no_toss_day_off = slot.no_toss_day_off;
            slot.env.open_rule = slot.open_rule;
            slot.env.merge_rule = slot.merge_rule;
            slot.steps = 0;
            slot.practice = 0;
            slot.aux = None;
            slot.aux_prep = false;
            slot.aux_hold = None;
        }
        Ok(())
    }

    /// 지금 상태의 관측과 마스크를 쓴다(게임을 바꾸지 않는다).
    fn observe(
        &self,
        mut grid: PyReadwriteArrayDyn<'_, f32>,
        mut scal: PyReadwriteArrayDyn<'_, f32>,
        mut mask: PyReadwriteArrayDyn<'_, bool>,
    ) -> PyResult<()> {
        let n = self.slots.len();
        let g = slice_mut(&mut grid, n * GRID_LEN, "grid")?;
        let s = slice_mut(&mut scal, n * N_SCALAR, "scal")?;
        let m = slice_mut(&mut mask, n * N_ACTIONS, "mask")?;
        for (((slot, g), s), m) in self.slots.iter().zip(g.chunks_mut(GRID_LEN)).zip(s.chunks_mut(N_SCALAR)).zip(m.chunks_mut(N_ACTIONS)) {
            slot.env.write_obs(g, s);
            slot.env.mask(m);
        }
        Ok(())
    }

    /// 게임 i의 요약: (day, hearts, swaps, 단계 코드, 점수)
    fn state(&self, i: usize) -> (i32, i32, i64, i32, i64) {
        let g = &self.slots[i].env.game;
        (g.day, g.hearts, g.swaps, g.phase.code(), g.score())
    }

    /// 게임 i의 보드 문자열 (1..7행 × 6열)과 포탑
    fn board(&self, i: usize) -> (Vec<Vec<String>>, Vec<String>) {
        let g = &self.slots[i].env.game;
        (g.board_cells(), g.turret_cells())
    }

    /// 게임 i의 지금 상태에서 드래그 행동(0..168)마다 복제본에 적용해 본 결과를 out[행동]에 쓴다(게임은 바뀌지 않는다).
    /// 열: 0..4 무기(화살탑·발리스타·대포) 등급 1..4 생성 수, 4 얼음벽 생성 수, 5 상자 합성(2등급 이상 상자 생성) 수,
    /// 6 버리기 여부, 7 성 보수로 얻은 하트. 무효한 드래그는 모든 열이 -1. 생성 수는 엔진 통계(연쇄 포함, 이동은 세지 않음)
    fn drag_outcomes(&self, i: usize, mut out: PyReadwriteArrayDyn<'_, i32>) -> PyResult<()> {
        let o = out.as_slice_mut().map_err(|e| PyValueError::new_err(format!("out: {e}")))?;
        if o.len() != A_DRAG_END * 8 {
            return Err(PyValueError::new_err(format!("out: 길이 {} (기대 {})", o.len(), A_DRAG_END * 8)));
        }
        let env = &self.slots[i].env;
        let g0 = &env.game;
        for a in 0..A_DRAG_END {
            let row = &mut o[a * 8..a * 8 + 8];
            let c = a / 4;
            let (x, y, dir) = ((c % 6) as i32 + 1, (c / 6) as i32 + 1, Dir::from_index(a % 4));
            let toss = g0.drag_is_toss(x, y, dir);
            let mut e = env.clone();
            if e.step(a).is_none() {
                row.fill(-1);
                continue;
            }
            let g = &e.game;
            for t in 1..=4 {
                row[t - 1] = (0..3).map(|k| (g.stat_made[k][t] - g0.stat_made[k][t]) as i32).sum();
            }
            row[4] = (1..=4).map(|t| (g.stat_made[3][t] - g0.stat_made[3][t]) as i32).sum();
            row[5] = (2..=4).map(|t| (g.chest_made[t] - g0.chest_made[t]) as i32).sum();
            row[6] = toss as i32;
            row[7] = if g.day == g0.day { (g.hearts - g0.hearts).max(0) } else { 0 };
        }
        Ok(())
    }

    /// 게임 i의 지금까지 만들어진 방어물: (종류 5 × 결과 등급 1..4, 종류 5 × 결과 행 0..7), 휴식일 번호
    fn made(&self, i: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, i32) {
        let g = &self.slots[i].env.game;
        (g.stat_made.iter().map(|r| r[1..].to_vec()).collect(), g.stat_made_row.iter().map(|r| r.to_vec()).collect(), g.day_off_day)
    }
}

/// 탐색 슬롯 하나: 복제한 게임과 그날 밤이 끝날 때까지의 기록
struct SearchSlot {
    env: Env,
    day0: i32,
    hearts0: i32,
    ret: f32,  // 누적 보상
    days: u32, // 지난 날 수
    active: bool,
}

impl SearchSlot {
    /// 행동 하나를 적용하고, 날이 바뀌었거나 게임이 끝났으면 멈춘다
    fn apply(&mut self, a: usize) -> Result<(), String> {
        let phase = self.env.game.phase;
        let o = self.env.step(a).ok_or_else(|| format!("탐색 슬롯: 무효 행동 {a} (단계 {phase:?})"))?;
        self.ret += o.reward;
        self.days += o.days;
        if o.done || self.env.game.day != self.day0 {
            self.active = false;
        }
        Ok(())
    }
}

/// 추론 시 탐색용 복제 환경: VecEnv의 게임을 슬롯마다 복제해(난수는 새 시드) 첫 행동을 적용하고,
/// 파이썬이 준 행동으로 그날 밤이 끝날 때까지(날이 바뀌거나 게임이 끝날 때까지) 진행한다. 끝난 슬롯은 멈추고
/// 관측 배열에는 끝 상태가 남는다(가치 추정용).
#[pyclass]
struct SearchEnv {
    slots: Vec<SearchSlot>,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl SearchEnv {
    /// m: 슬롯 수, threads: 0이면 CPU 수
    #[new]
    #[pyo3(signature = (m, threads = 0))]
    fn new(m: usize, threads: usize) -> PyResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let slots = (0..m)
            .map(|_| SearchSlot { env: Env::new(1, 1), day0: 0, hearts0: 0, ret: 0.0, days: 0, active: false })
            .collect();
        Ok(SearchEnv { slots, pool })
    }

    /// 앞쪽 k = len(idx)개 슬롯에 src의 게임 idx[j]를 복제하고 난수를 seeds[j]에서 만든 두 시드로 바꾼 뒤 first[j]를 적용한다
    /// (first[j] < 0이면 행동 없이 복제만). 나머지 슬롯은 멈춘다. grid [M, C, H, W], scal [M, S], mask [M, A], active [M]를 쓴다.
    #[allow(clippy::too_many_arguments)]
    fn fork(
        &mut self,
        py: Python<'_>,
        src: PyRef<'_, VecEnv>,
        idx: PyReadonlyArray1<'_, i64>,
        first: PyReadonlyArray1<'_, i64>,
        seeds: PyReadonlyArray1<'_, u64>,
        mut grid: PyReadwriteArrayDyn<'_, f32>,
        mut scal: PyReadwriteArrayDyn<'_, f32>,
        mut mask: PyReadwriteArrayDyn<'_, bool>,
        mut active: PyReadwriteArray1<'_, bool>,
    ) -> PyResult<()> {
        let m = self.slots.len();
        let (idx, first, seeds) = (idx.as_slice()?, first.as_slice()?, seeds.as_slice()?);
        let k = idx.len();
        if k > m || first.len() != k || seeds.len() != k {
            return Err(PyValueError::new_err(format!("fork: idx {k}, first {}, seeds {} (슬롯 {m})", first.len(), seeds.len())));
        }
        if let Some(&i) = idx.iter().find(|&&i| i < 0 || i as usize >= src.slots.len()) {
            return Err(PyValueError::new_err(format!("fork: 게임 번호 {i}")));
        }
        let g = slice_mut(&mut grid, m * GRID_LEN, "grid")?;
        let s = slice_mut(&mut scal, m * N_SCALAR, "scal")?;
        let mk = slice_mut(&mut mask, m * N_ACTIONS, "mask")?;
        let act = slice_mut(&mut active, m, "active")?;
        let (slots, pool, src_slots) = (&mut self.slots, &self.pool, &src.slots);
        let out: Result<(), String> = py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .enumerate()
                    .zip(g.par_chunks_mut(GRID_LEN))
                    .zip(s.par_chunks_mut(N_SCALAR))
                    .zip(mk.par_chunks_mut(N_ACTIONS))
                    .zip(act.par_iter_mut())
                    .try_for_each(|(((((j, slot), g), s), mk), act)| {
                        if j >= k {
                            slot.active = false;
                            *act = false;
                            return Ok(());
                        }
                        let mut env = src_slots[idx[j] as usize].env.clone();
                        let mut r = seeds[j];
                        env.game.rng_v = JsRng::new(splitmix(&mut r) as u32 | 1);
                        env.game.rng_m = JsRng::new(splitmix(&mut r) as u32 | 1);
                        *slot = SearchSlot { day0: env.game.day, hearts0: env.game.hearts, env, ret: 0.0, days: 0, active: true };
                        if first[j] >= 0 {
                            slot.apply(first[j] as usize)?;
                        }
                        slot.env.write_obs(g, s);
                        slot.env.mask(mk);
                        *act = slot.active;
                        Ok(())
                    })
            })
        });
        out.map_err(PyValueError::new_err)
    }

    /// src 게임 idx[j]를 슬롯 dst[j]에 행동 없이 복제하고 난수를 seeds[j]에서 만든 두 시드로 바꾼다. 다른 슬롯은 그대로 둔다
    /// (여러 시점의 상태를 모아 둘 때 쓴다).
    fn store(&mut self, src: PyRef<'_, VecEnv>, idx: PyReadonlyArray1<'_, i64>, dst: PyReadonlyArray1<'_, i64>, seeds: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let (idx, dst, seeds) = (idx.as_slice()?, dst.as_slice()?, seeds.as_slice()?);
        if idx.len() != dst.len() || idx.len() != seeds.len() {
            return Err(PyValueError::new_err("store: idx, dst, seeds 길이가 다르다"));
        }
        for ((&i, &d), &sd) in idx.iter().zip(dst).zip(seeds) {
            let (i, d) = (i as usize, d as usize);
            if i >= src.slots.len() || d >= self.slots.len() {
                return Err(PyValueError::new_err(format!("store: 게임 {i} 또는 슬롯 {d} 범위 밖")));
            }
            let mut env = src.slots[i].env.clone();
            let mut r = sd;
            env.game.rng_v = JsRng::new(splitmix(&mut r) as u32 | 1);
            env.game.rng_m = JsRng::new(splitmix(&mut r) as u32 | 1);
            self.slots[d] = SearchSlot { day0: env.game.day, hearts0: env.game.hearts, env, ret: 0.0, days: 0, active: false };
        }
        Ok(())
    }

    /// 진행 중인 슬롯에만 행동을 적용한다(멈춘 슬롯의 행동은 무시). grid, scal, mask, active를 갱신한다.
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<'_, i64>,
        mut grid: PyReadwriteArrayDyn<'_, f32>,
        mut scal: PyReadwriteArrayDyn<'_, f32>,
        mut mask: PyReadwriteArrayDyn<'_, bool>,
        mut active: PyReadwriteArray1<'_, bool>,
    ) -> PyResult<()> {
        let m = self.slots.len();
        let acts = actions.as_slice()?;
        if acts.len() != m {
            return Err(PyValueError::new_err(format!("actions: 길이 {} (기대 {m})", acts.len())));
        }
        let g = slice_mut(&mut grid, m * GRID_LEN, "grid")?;
        let s = slice_mut(&mut scal, m * N_SCALAR, "scal")?;
        let mk = slice_mut(&mut mask, m * N_ACTIONS, "mask")?;
        let act = slice_mut(&mut active, m, "active")?;
        let (slots, pool) = (&mut self.slots, &self.pool);
        let out: Result<(), String> = py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .zip(g.par_chunks_mut(GRID_LEN))
                    .zip(s.par_chunks_mut(N_SCALAR))
                    .zip(mk.par_chunks_mut(N_ACTIONS))
                    .zip(act.par_iter_mut())
                    .zip(acts.par_iter())
                    .try_for_each(|(((((slot, g), s), mk), act), &a)| {
                        if !slot.active {
                            *act = false;
                            return Ok(());
                        }
                        slot.apply(a as usize)?;
                        slot.env.write_obs(g, s);
                        slot.env.mask(mk);
                        *act = slot.active;
                        Ok(())
                    })
            })
        });
        out.map_err(PyValueError::new_err)
    }

    /// 슬롯 j의 보드 문자열과 포탑
    fn board(&self, j: usize) -> (Vec<Vec<String>>, Vec<String>) {
        let g = &self.slots[j].env.game;
        (g.board_cells(), g.turret_cells())
    }

    /// 슬롯별 결과: 누적 보상, 지난 날 수, 사망 여부, 잃은 하트(사망이면 복제 시점 하트 전부)
    fn results(
        &self,
        mut ret: PyReadwriteArray1<'_, f32>,
        mut days: PyReadwriteArray1<'_, f32>,
        mut dead: PyReadwriteArray1<'_, bool>,
        mut lost: PyReadwriteArray1<'_, f32>,
    ) -> PyResult<()> {
        let m = self.slots.len();
        let (r, d, de, l) = (
            slice_mut(&mut ret, m, "ret")?,
            slice_mut(&mut days, m, "days")?,
            slice_mut(&mut dead, m, "dead")?,
            slice_mut(&mut lost, m, "lost")?,
        );
        for (j, slot) in self.slots.iter().enumerate() {
            let over = slot.env.game.phase == Phase::GameOver;
            r[j] = slot.ret;
            d[j] = slot.days as f32;
            de[j] = over;
            l[j] = if over { slot.hearts0 as f32 } else { (slot.hearts0 - slot.env.game.hearts) as f32 };
        }
        Ok(())
    }
}

#[pymodule]
fn towerswap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<SearchEnv>()?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    m.add("N_SCALAR", N_SCALAR)?;
    m.add("GRID_SHAPE", (N_GRID_CH, GRID_H, GRID_W))?;
    Ok(())
}
