// 상자·아이템·탭 행동을 포함해 원본과 Rust 엔진을 대조한다(스크립트 모드: 정수 r을 양쪽이 같은 규칙으로 해석).
// 사용: node compare_items.js [케이스 수] [케이스당 행동 수] [시작 시드]
"use strict";
const { execFileSync } = require("child_process");
const path = require("path");
const { Oracle } = require("./driver.js");

const RUST_CLI = path.join(__dirname, "..", "engine", "target", "release", "towerswap-cli");
const N = +(process.argv[2] || 40);
const ACTIONS = +(process.argv[3] || 60);
let seed = +(process.argv[4] || 1);
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648);
const pick = (a) => a[Math.floor(rnd() * a.length)];
const int = (a, b) => a + Math.floor(rnd() * (b - a + 1));

function randomCell() {
  const r = rnd();
  if (r < 0.40) return pick(["s", "l", "i", "g", "d"]) + "0";
  if (r < 0.62) return pick(["t", "b", "c", "w"]) + int(1, 4);
  if (r < 0.80) return "h" + int(1, 4);
  if (r < 0.85) return "a" + int(1, 4);
  if (r < 0.89) return "T" + int(1, 3);
  if (r < 0.93) return "F0";
  if (r < 0.95) return "H2";
  if (r < 0.97) return pick(["C", "D"]) + int(0, 3);
  return "e0";
}
function randomBoard() {
  let s = "";
  for (let x = 1; x <= 6; x++) s += (x === 1 || x === 6) && rnd() < 0.3 ? "t" + int(1, 3) : "e0";
  for (let y = 1; y <= 6; y++) for (let x = 1; x <= 6; x++) s += randomCell();
  for (let x = 1; x <= 6; x++) s += rnd() < 0.2 ? "I" + int(1, 3) : "o0";
  return s;
}

const FIELDS = ["tK", "day", "hearts", "swaps", "sN", "bossCol", "dayOff", "deals", "declined", "vseed", "mseed"];
function diff(a, b) {
  const over = (s) => [5, 11, 28].includes(s.tK);
  if (over(a) || over(b)) return over(a) && over(b) && a.day === b.day ? null : `게임오버: 원본 tK=${a.tK} day=${a.day} rust tK=${b.tK} day=${b.day}`;
  for (const f of FIELDS) if (a[f] !== b[f]) return `${f}: 원본=${a[f]} rust=${b[f]}`;
  if (JSON.stringify(a.board) !== JSON.stringify(b.board)) return "board";
  if (JSON.stringify(a.turrets) !== JSON.stringify(b.turrets)) return "turrets";
  return null;
}
const show = (s) => `day=${s.day} hearts=${s.hearts} swaps=${s.swaps} tK=${s.tK} deals=${s.deals}/${s.declined}\n` +
  s.board.map((r) => "    " + r.map((c) => c.padEnd(3)).join(" ")).join("\n") + "\n    turrets " + JSON.stringify(s.turrets);

let pass = 0, fail = 0, compared = 0;
const seenStates = new Map();
for (let c = 0; c < N; c++) {
  const vseed = int(1, 2147483000), mseed = int(1, 2147483000);
  let day = int(1, 45);
  if (day % 10 === 0) day += 1;
  const setup = { day, hearts: int(2, 36), achievements: Math.min(5, Math.floor((day - 1) / 10)) };
  const board = randomBoard();
  const actions = [];
  for (let i = 0; i < ACTIONS; i++) actions.push({ r: int(0, 1 << 30) });
  const rust = JSON.parse(execFileSync(RUST_CLI, { input: JSON.stringify({ vseed, mseed, board, setup, actions }), maxBuffer: 1 << 28 }).toString()).states;

  const o = new Oracle();
  o.stopAtDusk = true;
  o.newGame(vseed, mseed);
  o.ev(`(function(s,d,h,a){ lR(s); tK=0; nq=null; nZ=null; r7=5; rg=0; tr=d; il=h; i_=h; sN=a; })`)(board, setup.day, setup.hearts, setup.achievements);
  o.frame();
  let orig = o.state();
  let bad = null;
  for (let i = 0; i < rust.length; i++) {
    const d = diff(orig, rust[i]);
    if (d) { bad = { i, d, orig, r: rust[i], prev: i > 0 ? rust[i - 1] : null }; break; }
    compared++;
    seenStates.set(orig.tK, (seenStates.get(orig.tK) || 0) + 1);
    if (i === actions.length || [5, 11, 28].includes(orig.tK)) break;
    o.interpret(actions[i].r);
    orig = o.state();
  }
  if (bad) {
    fail++;
    console.log(`✗ case ${c} 행동 ${bad.i}에서 불일치: ${bad.d} (setup=${JSON.stringify(setup)})`);
    if (fail <= 2) {
      console.log(`  board=${board}`);
      if (bad.prev) console.log(`  직전 r=${actions[bad.i - 1].r}, 직전 상태(rust): ` + show(bad.prev));
      console.log("  원본: " + show(bad.orig) + "\n  rust: " + show(bad.r));
    }
  } else pass++;
}
console.log(`결과: 통과 ${pass} / 실패 ${fail} (비교한 상태 ${compared}개)`);
console.log("거친 상태 분포(tK→횟수):", JSON.stringify([...seenStates.entries()].sort((a, b) => a[0] - b[0])));
process.exit(fail ? 1 : 0);
