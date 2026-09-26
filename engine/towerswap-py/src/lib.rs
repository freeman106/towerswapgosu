//! 파이썬 바인딩: N개 게임을 rayon 스레드로 병렬 진행하는 배치 환경.
//! 관측·마스크·보상은 파이썬이 미리 할당한 numpy 배열에 바로 쓴다(복사 없음).
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use towerswap_core::analysis::Board;
use towerswap_core::env::{Env, GRID_H, GRID_LEN, GRID_W, N_ACTIONS, N_GRID_CH, N_SCALAR};
use towerswap_core::expert::{greedy_set, greedy_set_with, merge_not_worse, Params};
use towerswap_core::kinds::Kind;
use towerswap_core::rng::JsRng;
use towerswap_core::{Game, Phase};

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
    practice: bool,           // 지금 게임이 연습 시작 상태에서 시작했는지
    aux: Option<Vec<bool>>,   // 지금 상태가 보조 손실 대상이면 그 합성 행동 집합 M(s)
}

/// 연습 시작 상태. `merge`는 합성이 유리하다고 확인된 경우의 합성 행동 집합 M(s)
struct Start {
    game: Game,
    merge: Option<Vec<bool>>,
}

impl Slot {
    /// 새 게임. 확률 `frac`로 연습 시작 풀의 상태에서 시작한다(난수는 새로 뽑고 통계는 비운다).
    fn new_game(&mut self, starts: &[Start], frac: f64) {
        let v = splitmix(&mut self.rng) as u32 | 1; // 시드 0은 허용하지 않는다
        let m = splitmix(&mut self.rng) as u32 | 1;
        let r = splitmix(&mut self.rng);
        self.practice = !starts.is_empty() && ((r >> 11) as f64 / (1u64 << 53) as f64) < frac;
        self.aux = None;
        self.env = if self.practice {
            let st = &starts[(splitmix(&mut self.rng) % starts.len() as u64) as usize];
            self.aux = st.merge.clone();
            let mut g = st.game.clone();
            g.rng_v = JsRng::new(v);
            g.rng_m = JsRng::new(m);
            g.reset_stats();
            Env::from_game(g)
        } else {
            Env::new(v, m)
        };
        self.env.chest_bonus = self.chest_bonus;
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

/// 합성 대신 상자를 여는 행동: 합성되는 등급의 상자 탭 중 첫 번째
fn open_instead(env: &Env, merge: &[bool]) -> Option<usize> {
    let g = &env.game;
    let mut m = [false; N_ACTIONS];
    env.mask(&mut m);
    let a = merge.iter().position(|&x| x)?;
    let mut e = env.clone();
    e.step(a).unwrap();
    let tier = (2..=4).find(|&t| e.game.chest_made[t] > g.chest_made[t])? as u8 - 1;
    (towerswap_core::env::A_CELL..towerswap_core::env::A_YES).find(|&a| {
        let c = a - towerswap_core::env::A_CELL;
        m[a] && g.grid[c % 6 + 1][c / 6].map_or(false, |t| g.tiles[t].kind == Kind::Chest && g.tiles[t].tier == tier)
    })
}

/// 상자를 모으는 greedy로 게임을 두어 조건 `level`에 맞는 상태를 하나 찾는다
fn find_start(level: u32, seed: u64) -> Game {
    let (mut rng, p) = (seed, Params::hoarding());
    let (mut m, mut set) = ([false; N_ACTIONS], [false; N_ACTIONS]);
    loop {
        let mut env = Env::new(splitmix(&mut rng) as u32 | 1, splitmix(&mut rng) as u32 | 1);
        while !env.done() && env.game.day < 60 {
            // 맞는 상태를 절반 확률로 받는다(게임 안의 여러 시점이 고루 뽑히도록)
            if start_level(&env.game) == Some(level) && splitmix(&mut rng) % 2 == 0 {
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
    made: [u32; 4],   // 등급 1..4 상자 생성 수
    opened: [u32; 4], // 등급 1..4 상자 개봉 수
    hold: [(u64, u64); 4], // 등급 1..4 개봉한 상자의 보관 기간 합 (행동 수, 날)
    merge_opps: u32,
    merge_taken: u32,
    max_normal_held: u32,
    practice: bool,
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
            made: g.chest_made[1..].try_into().unwrap(),
            opened: g.chest_opened[1..].try_into().unwrap(),
            hold: g.chest_hold[1..].try_into().unwrap(),
            merge_opps: e.merge_opps,
            merge_taken: e.merge_taken,
            max_normal_held: e.max_normal_held,
            practice: slot.practice,
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
    /// chest_bonus: 등급 1..4 상자를 게임에서 처음 만들 때의 보상 [b1, b2, b3, b4]
    #[new]
    #[pyo3(signature = (n, seed = 1, threads = 0, max_steps = 200_000, chest_bonus = None))]
    fn new(n: usize, seed: u64, threads: usize, max_steps: u64, chest_bonus: Option<Vec<f32>>) -> PyResult<Self> {
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
                    practice: false,
                    aux: None,
                };
                s.new_game(&[], 0.0);
                s
            })
            .collect();
        Ok(VecEnv { slots, pool, max_steps, finished: Vec::new(), starts: Vec::new(), start_frac: 0.0 })
    }

    /// 연습 시작 상태 `count`개를 만든다(상자를 모으는 greedy의 실제 게임에서). 기존 풀은 바꾼다.
    /// level 1: 교환 한 번이면 상자 합성, 2: 교환 두 번, 3: 같은 등급 상자 3개 이상, 4: 같은 등급 상자 2개
    /// confirm_samples > 0이면(난이도 1) 상태마다 합성과 상자 개봉을 그날 밤까지 굴려 비교하고(시드 쌍 수),
    /// 합성이 나쁘지 않은 상태에만 합성 행동 집합 M(s)을 붙인다(보조 손실 대상).
    #[pyo3(signature = (count, level, seed = 1, confirm_samples = 0))]
    fn build_start_pool(&mut self, py: Python<'_>, count: usize, level: u32, seed: u64, confirm_samples: u32) -> PyResult<()> {
        if !(1..=4).contains(&level) {
            return Err(PyValueError::new_err("level: 1..4"));
        }
        let pool = &self.pool;
        self.starts = py.detach(|| {
            pool.install(|| {
                (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let mut rng = splitmix(&mut (seed ^ (i as u64) << 20));
                        let game = find_start(level, rng);
                        let mut merge = None;
                        if level == 1 && confirm_samples > 0 {
                            let env = Env::from_game(game.clone());
                            let m = merge_set(&env);
                            if let (Some(a), Some(open)) = (m.iter().position(|&x| x), open_instead(&env, &m)) {
                                if merge_not_worse(&env, a, open, confirm_samples, &mut rng) {
                                    merge = Some(m);
                                }
                            }
                        }
                        Start { game, merge }
                    })
                    .collect()
            })
        });
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
                        slot.aux = None;
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
    /// score, day, steps, truncated, env, max_chest, r_survival, r_boss, r_chest(보상 성분 합계),
    /// made, opened(등급 1..4 상자 생성·개봉 수), hold_steps, hold_days(등급 1..4 개봉한 상자의 보관 기간 합),
    /// merge_opps, merge_taken(교환 한 번 상자 합성 기회와 실제 합성), max_normal_held, practice(연습 시작 여부)
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
                d.set_item("made", f.made.to_vec())?;
                d.set_item("opened", f.opened.to_vec())?;
                d.set_item("hold_steps", f.hold.iter().map(|h| h.0).collect::<Vec<_>>())?;
                d.set_item("hold_days", f.hold.iter().map(|h| h.1).collect::<Vec<_>>())?;
                d.set_item("merge_opps", f.merge_opps)?;
                d.set_item("merge_taken", f.merge_taken)?;
                d.set_item("max_normal_held", f.max_normal_held)?;
                d.set_item("practice", f.practice)?;
                Ok(d)
            })
            .collect()
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
}

#[pymodule]
fn towerswap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    m.add("N_SCALAR", N_SCALAR)?;
    m.add("GRID_SHAPE", (N_GRID_CH, GRID_H, GRID_W))?;
    Ok(())
}
