// 밤을 포함한 전체 게임을 원본과 Rust 엔진으로 대조한다.
// 무작위 강한 보드(타워 다수)와 넉넉한 하트로 여러 날을 넘기게 한다.
// 사용: node compare_full.js [케이스 수] [케이스당 행동 수] [시작 시드]
"use strict";
const { execFileSync } = require("child_process");
const path = require("path");
const { Oracle } = require("./driver.js");

const RUST_CLI = path.join(__dirname, "..", "engine", "target", "release", "towerswap-cli");
const N = +(process.argv[2] || 30);
const ACTIONS = +(process.argv[3] || 60);
let seed = +(process.argv[4] || 1);
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648);
const pick = (a) => a[Math.floor(rnd() * a.length)];
const int = (a, b) => a + Math.floor(rnd() * (b - a + 1));

function randomCell() {
  const r = rnd();
  if (r < 0.45) return pick(["s", "l", "i", "g", "d"]) + "0";
  if (r < 0.9) return pick(["t", "b", "c", "w", "t", "b", "c"]) + int(1, 4);
  if (r < 0.95) return "f" + int(1, 4);
  return pick(["C", "D"]) + int(1, 4);
}
function randomBoard() {
  let s = "";
  for (let x = 1; x <= 6; x++) s += (x === 1 || x === 6) && rnd() < 0.5 ? "t" + int(1, 4) : "e0";
  for (let y = 1; y <= 6; y++) for (let x = 1; x <= 6; x++) s += randomCell();
  for (let x = 1; x <= 6; x++) s += rnd() < 0.25 ? "I" + int(1, 3) : "o0";
  return s;
}

const FIELDS = ["tK", "day", "hearts", "swaps", "sN", "bossCol", "dayOff", "vseed", "mseed"];
function diff(a, b) {
  // 게임 오버는 같은 날이면 동일하게 본다(원본은 같은 프레임의 남은 드래곤이 하트를 음수로 더 깎는다)
  const over = (s) => [5, 11, 28].includes(s.tK);
  if (over(a) || over(b)) return over(a) && over(b) && a.day === b.day ? null : `게임오버: 원본 tK=${a.tK} day=${a.day} rust tK=${b.tK} day=${b.day}`;
  for (const f of FIELDS) if (a[f] !== b[f]) return `${f}: 원본=${a[f]} rust=${b[f]}`;
  if (JSON.stringify(a.board) !== JSON.stringify(b.board)) return "board";
  if (JSON.stringify(a.turrets) !== JSON.stringify(b.turrets)) return "turrets";
  return null;
}
const show = (s) => `day=${s.day} hearts=${s.hearts} swaps=${s.swaps} tK=${s.tK} boss=${s.bossCol} sN=${s.sN}\n` +
  s.board.map((r) => "    " + r.map((c) => c.padEnd(3)).join(" ")).join("\n");

let pass = 0, fail = 0, compared = 0, nights = 0, maxDay = 0, gameOvers = 0;
for (let c = 0; c < N; c++) {
  const vseed = int(1, 2147483000), mseed = int(1, 2147483000);
  let day = int(1, 58);
  if (day % 10 === 0) day += 1; // 보스 날 직접 시작은 피한다(보스 열은 그날 아침에 정해짐)
  const setup = { day, hearts: int(3, 36), achievements: Math.min(5, Math.floor((day - 1) / 10)) };
  const board = randomBoard();
  const dayoff = rnd() < 0.5;
  const actions = [];
  for (let i = 0; i < ACTIONS; i++) actions.push({ drag: [int(1, 6), int(1, 7), int(0, 3)] });
  const rust = JSON.parse(execFileSync(RUST_CLI, { input: JSON.stringify({ vseed, mseed, board, setup, dayoff, actions }) }).toString()).states;

  const o = new Oracle();
  o.newGame(vseed, mseed);
  o.ev(`(function(s,d,h,a){ lR(s); tK=0; nq=null; nZ=null; r7=5; rg=0; tr=d; il=h; i_=h; sN=a; })`)(board, setup.day, setup.hearts, setup.achievements);
  o.frame();
  let orig = o.state();
  let bad = null;
  for (let i = 0; i < rust.length; i++) {
    const d = diff(orig, rust[i]);
    if (d) { bad = { i, d, orig, r: rust[i], prev: i > 0 ? rust[i - 1] : null }; break; }
    compared++;
    maxDay = Math.max(maxDay, orig.day);
    if (i === actions.length || [5, 11, 28].includes(orig.tK)) break;
    const dayBefore = orig.day;
    o.drag(...actions[i].drag);
    while (o.ev("tK") === 108) { o.ev(dayoff ? "_D(eU = tr, 'dayoffday'), _a(1)" : "_a(1)"); o.runUntilInput(); }
    orig = o.state();
    if (orig.day !== dayBefore || [5, 11, 28].includes(orig.tK)) nights++;
    if ([5, 11, 28].includes(orig.tK)) gameOvers++;
  }
  if (bad) {
    fail++;
    console.log(`✗ case ${c}: ${bad.d} (setup=${JSON.stringify(setup)})`);
    if (fail <= 0) {
      console.log(`✗ case ${c} 행동 ${bad.i}에서 불일치: ${bad.d}\n  setup=${JSON.stringify(setup)} dayoff=${dayoff}\n  board=${board}`);
      if (bad.prev) console.log("  직전 행동:", JSON.stringify(actions[bad.i - 1]), "\n  직전 상태(rust): " + show(bad.prev));
      console.log("  원본: " + show(bad.orig) + "\n  rust: " + show(bad.r));
    }
  } else pass++;
}
console.log(`결과: 통과 ${pass} / 실패 ${fail} (비교한 상태 ${compared}개, 밤 ${nights}회, 게임오버 ${gameOvers}회, 최대 day ${maxDay})`);
process.exit(fail ? 1 : 0);
