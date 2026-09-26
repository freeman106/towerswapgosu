// 원본(oracle)과 Rust 엔진을 같은 시드·행동으로 돌려 상태를 비교한다.
// 사용: node compare.js [케이스 수] [케이스당 행동 수] [시작 시드]
"use strict";
const { execFileSync } = require("child_process");
const path = require("path");
const { Oracle } = require("./driver.js");

const RUST_CLI = path.join(__dirname, "..", "engine", "target", "release", "towerswap-cli");
const N = +(process.argv[2] || 50);
const ACTIONS = +(process.argv[3] || 30);
let seed = +(process.argv[4] || 1);

function lcg() {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
}

// 비교할 필드 (loops는 내부 시각이라 제외)
const FIELDS = ["tK", "day", "hearts", "swaps", "sN", "bossCol", "dayOff", "deals", "declined", "vseed", "mseed"];

function diff(a, b) {
  for (const f of FIELDS) if (a[f] !== b[f]) return `${f}: 원본=${a[f]} rust=${b[f]}`;
  if (JSON.stringify(a.board) !== JSON.stringify(b.board)) return "board";
  if (JSON.stringify(a.turrets) !== JSON.stringify(b.turrets)) return "turrets";
  return null;
}

function show(s) {
  return s.board.map((r) => r.map((c) => c.padEnd(3)).join(" ")).join("\n");
}

let pass = 0, fail = 0, comparedActions = 0;
for (let c = 0; c < N; c++) {
  const vseed = 1 + Math.floor(lcg() * 2147483000);
  const mseed = 1 + Math.floor(lcg() * 2147483000);
  const actions = [];
  for (let i = 0; i < ACTIONS; i++) {
    actions.push({ drag: [1 + Math.floor(lcg() * 6), 1 + Math.floor(lcg() * 6), Math.floor(lcg() * 4)] });
  }
  const input = JSON.stringify({ vseed, mseed, actions });
  const rust = JSON.parse(execFileSync(RUST_CLI, { input }).toString()).states;

  // 원본은 행동마다 직접 돌린다 (밤이 시작되면 Rust가 아직 밤을 지원하지 않으므로 거기서 멈춘다)
  const o = new Oracle();
  o.newGame(vseed, mseed);
  let orig = o.state();
  let bad = null;
  for (let i = 0; i <= actions.length && i < rust.length; i++) {
    const r = rust[i];
    if (r.tK === 25) break; // Rust 쪽 밤 직전 → 여기까지만 비교
    const d = diff(orig, r);
    if (d) { bad = { i, d, orig, r }; break; }
    comparedActions++;
    if (i === actions.length) break;
    o.drag(...actions[i].drag);
    orig = o.state();
  }
  if (bad) {
    fail++;
    if (fail <= 3) {
      console.log(`✗ case ${c} (vseed=${vseed}, mseed=${mseed}) 행동 ${bad.i}에서 불일치: ${bad.d}`);
      if (bad.i > 0) console.log("  직전 행동:", JSON.stringify(actions[bad.i - 1]));
      console.log("  원본:\n" + show(bad.orig) + "\n  rust:\n" + show(bad.r));
    }
  } else pass++;
}
console.log(`결과: 통과 ${pass} / 실패 ${fail} (비교한 상태 ${comparedActions}개)`);
process.exit(fail ? 1 : 0);
