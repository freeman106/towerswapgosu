//! 파이썬 바인딩: N개 게임을 rayon 스레드로 병렬 진행하는 배치 환경.
//! 관측·마스크·보상은 파이썬이 미리 할당한 numpy 배열에 바로 쓴다(복사 없음).
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use towerswap_core::env::{Env, GRID_H, GRID_LEN, GRID_W, N_ACTIONS, N_GRID_CH, N_SCALAR};

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
}

impl Slot {
    fn new_game(&mut self) {
        let v = splitmix(&mut self.rng) as u32 | 1; // 시드 0은 허용하지 않는다
        let m = splitmix(&mut self.rng) as u32 | 1;
        self.env = Env::new(v, m);
        self.steps = 0;
    }
}

/// 끝난 게임 기록: (점수, 도달한 날, 스텝 수, 스텝 상한으로 잘림)
type Finished = (i64, i32, u64, bool);

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
}

#[pymethods]
impl VecEnv {
    /// n: 게임 수, seed: 시드, threads: 0이면 CPU 수, max_steps: 한 게임의 스텝 상한(안전장치)
    #[new]
    #[pyo3(signature = (n, seed = 1, threads = 0, max_steps = 200_000))]
    fn new(n: usize, seed: u64, threads: usize, max_steps: u64) -> PyResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut root = seed;
        let slots = (0..n)
            .map(|_| {
                let mut s = Slot { env: Env::new(1, 1), rng: splitmix(&mut root), steps: 0 };
                s.new_game();
                s
            })
            .collect();
        Ok(VecEnv { slots, pool, max_steps, finished: Vec::new() })
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
        let (slots, pool) = (&mut self.slots, &self.pool);
        py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .zip(g.par_chunks_mut(GRID_LEN))
                    .zip(s.par_chunks_mut(N_SCALAR))
                    .zip(m.par_chunks_mut(N_ACTIONS))
                    .for_each(|(((slot, g), s), m)| {
                        slot.new_game();
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
        let (slots, pool, max_steps) = (&mut self.slots, &self.pool, self.max_steps);
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
                        let o = slot.env.step(a as usize).ok_or_else(|| format!("환경 {i}: 무효 행동 {a} (단계 {phase:?})"))?;
                        slot.steps += 1;
                        *r = o.reward;
                        *dy = o.days as f32;
                        let truncated = !o.done && slot.steps >= max_steps;
                        let fin = (o.done || truncated).then(|| {
                            let g = &slot.env.game;
                            let f = (g.score(), g.day, slot.steps, truncated);
                            slot.new_game();
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

    /// 지난 호출 이후 끝난 게임 기록을 꺼낸다: [(점수, 날, 스텝, 잘림)]
    fn pop_finished(&mut self) -> Vec<Finished> {
        std::mem::take(&mut self.finished)
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
