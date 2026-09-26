// 무작위 보드(빙산·슬롯 대포·모루·TNT·요정·구멍 포함)에서 원본과 Rust 엔진의 드래그 결과를 대조한다.
// 사용: node compare_boards.js [케이스 수] [케이스당 행동 수] [시작 시드]
"use strict";
const { execFileSync } = require("child_process");
const path = require("path");
const { Oracle } = require("./driver.js");

const RUST_CLI = path.join(__dirname, "..", "engine", "target", "release", "towerswap-cli");
const N = +(process.argv[2] || 100);
const ACTIONS = +(process.argv[3] || 20);
let seed = +(process.argv[4] || 1);
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648);
const pick = (a) => a[Math.floor(rnd() * a.length)];
const int = (a, b) => a + Math.floor(rnd() * (b - a + 1));

function randomCell() {
  const r = rnd();
  if (r < 0.55) return pick(["s", "l", "i", "g", "d"]) + "0";
  if (r < 0.72) return pick(["t", "b", "c", "w"]) + int(1, 4);
  if (r < 0.76) return "f" + int(1, 3);
  if (r < 0.81) return "h" + int(1, 3);
  if (r < 0.86) return "a" + int(1, 3);
  if (r < 0.89) return "T" + int(1, 3);
  if (r < 0.93) return pick(["C", "D"]) + int(0, 3);
  if (r < 0.96) return "e0";
  return "F0";
}

function randomBoard() {
  let s = "";
  for (let x = 1; x <= 6; x++) s += (x === 1 || x === 6) && rnd() < 0.3 ? "t" + int(1, 3) : "e0";
  for (let y = 1; y <= 6; y++) for (let x = 1; x <= 6; x++) s += randomCell();
  for (let x = 1; x <= 6; x++) s += rnd() < 0.35 ? "I" + int(1, 3) : "o0";
  return s;
}

const FIELDS = ["tK", "day", "hearts", "swaps", "vseed", "mseed"];
function diff(a, b) {
  for (const f of FIELDS) if (a[f] !== b[f]) return `${f}: 원본=${a[f]} rust=${b[f]}`;
  if (JSON.stringify(a.board) !== JSON.stringify(b.board)) return "board";
  if (JSON.stringify(a.turrets) !== JSON.stringify(b.turrets)) return "turrets";
  return null;
}
const show = (s) => s.board.map((r) => r.map((c) => c.padEnd(3)).join(" ")).join("\n") + "\n  turrets " + JSON.stringify(s.turrets);

let pass = 0, fail = 0, compared = 0;
for (let c = 0; c < N; c++) {
  const vseed = int(1, 2147483000), mseed = int(1, 2147483000);
  const board = randomBoard();
  const actions = [];
  for (let i = 0; i < ACTIONS; i++) actions.push({ drag: [int(1, 6), int(1, 7), int(0, 3)] });
  const rust = JSON.parse(execFileSync(RUST_CLI, { input: JSON.stringify({ vseed, mseed, board, actions }) }).toString()).states;

  const o = new Oracle();
  o.newGame(vseed, mseed);
  o.ev(`(function(s){ lR(s); tK=0; nq=null; nZ=null; r7=5; rg=0; })`)(board);
  o.frame(); // 실제 게임처럼 입력은 불러온 프레임보다 뒤에 처리된다
  let orig = o.state();
  let bad = null;
  for (let i = 0; i < rust.length; i++) {
    if (rust[i].tK === 25) break;
    const d = diff(orig, rust[i]);
    if (d) { bad = { i, d, orig, r: rust[i], prev: i > 0 ? rust[i - 1] : null }; break; }
    compared++;
    if (i === actions.length) break;
    o.drag(...actions[i].drag);
    orig = o.state();
  }
  if (bad) {
    fail++;
    if (fail <= 3) {
      console.log(`✗ case ${c} 행동 ${bad.i}에서 불일치: ${bad.d}\n  board=${board}`);
      if (bad.i > 0) console.log("  직전 행동:", JSON.stringify(actions[bad.i - 1]), "\n  직전 상태(rust):\n" + show(bad.prev));
      console.log("  원본:\n" + show(bad.orig) + "\n  rust:\n" + show(bad.r));
    }
  } else pass++;
}
console.log(`결과: 통과 ${pass} / 실패 ${fail} (비교한 상태 ${compared}개)`);
process.exit(fail ? 1 : 0);
