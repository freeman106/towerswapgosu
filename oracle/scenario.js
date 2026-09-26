// 지정한 보드에서 원본 동작을 재현하는 시나리오 검사.
// 보드 문자열은 원본 lR() 형식: 포탑 6칸 + 1..7행 × 6칸, 칸마다 (id, tier) 2글자.
"use strict";
const { Oracle, printState } = require("./driver.js");

const KINDS = ["s", "l", "i", "g", "d"];
// 3개가 한 줄로 이어지지 않는 기본 배치
function baseRows() {
  const rows = [];
  for (let y = 1; y <= 6; y++) {
    let r = "";
    for (let x = 1; x <= 6; x++) r += KINDS[(x + 2 * y) % 5] + "0";
    rows.push(r);
  }
  return rows;
}

function setup(o, rows, row7 = "o0o0o0o0o0o0", turrets = "e0e0e0e0e0e0") {
  o.newGame(1234, 5678);
  o.ev(`(function(s){ lR(s); tK=0; nq=null; nZ=null; r7=5; rg=0; })`)(turrets + rows.join("") + row7);
  o.frame(); // 실제 게임처럼 입력은 불러온 프레임보다 뒤에 처리된다
  return o.state();
}

function cell(s, x, y) {
  return s.board[y - 1][x - 1];
}

const results = [];
function check(name, cond, detail) {
  results.push({ name, pass: !!cond });
  console.log(`${cond ? "PASS" : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
}

// 1) 물 칸의 빙산 쪽으로 일반 타일을 끌면 버리기가 된다
{
  const o = new Oracle();
  const before = setup(o, baseRows(), "o0o0I1o0o0o0");
  const t36 = cell(before, 3, 6);
  o.drag(3, 6, 1);
  const after = o.state();
  check("iceberg-does-not-block-land-toss",
    after.swaps === 4 && cell(after, 3, 7) === "I1" && after.vseed !== before.vseed,
    `(3,6) ${t36} → 버림, 스왑 ${before.swaps}→${after.swaps}, (3,7)=${cell(after, 3, 7)}`);
  // 빙산을 땅 타일 쪽으로 끌면 무효
  const o2 = new Oracle();
  const b2 = setup(o2, baseRows(), "o0o0I1o0o0o0");
  o2.drag(3, 7, 0);
  const a2 = o2.state();
  check("iceberg-to-land-invalid", a2.swaps === b2.swaps && JSON.stringify(a2.board) === JSON.stringify(b2.board));
}

// 2) 대포 슬롯 위 대포를 버리면 슬롯이 비고, 위 타일은 내려오지 않으며 보드 난수도 쓰지 않는다
{
  const o = new Oracle();
  const rows = baseRows();
  rows[5] = "C1" + rows[5].slice(2); // (1,6)에 슬롯+대포
  const before = setup(o, rows);
  const t15 = cell(before, 1, 5);
  o.drag(1, 6, 2);
  const after = o.state();
  check("cannon-slot-does-not-refill",
    cell(after, 1, 6) === ".." && cell(after, 1, 5) === t15 && after.vseed === before.vseed && after.swaps === 4,
    `(1,6)=${cell(after, 1, 6)}, (1,5) ${t15}→${cell(after, 1, 5)}, vseed 변화=${after.vseed !== before.vseed}`);
}

// 3) 7행 빙산끼리(등급 다름) 교환 가능, 빙산을 보드 밖으로 버리기 가능
{
  const o = new Oracle();
  const before = setup(o, baseRows(), "I1I2o0o0o0o0");
  o.drag(1, 7, 3);
  const after = o.state();
  check("row7-icebergs-swap", cell(after, 1, 7) === "I2" && cell(after, 2, 7) === "I1" && after.swaps === 4);
  o.drag(1, 7, 2);
  const after2 = o.state();
  check("row7-iceberg-toss-out", cell(after2, 1, 7) === "~~" && after2.swaps === 3);
}

const fails = results.filter((r) => !r.pass).length;
console.log(`\n${results.length - fails}/${results.length} 통과`);
process.exit(fails ? 1 : 0);
