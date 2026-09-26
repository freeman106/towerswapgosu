// compare_full.js의 특정 케이스를 재현해, 불일치가 난 밤을 프레임 단위로 원본과 Rust에서 기록·비교한다.
// 사용: node debug_night.js <시작 시드> <케이스 번호> [행동 수]
"use strict";
const { execFileSync } = require("child_process");
const path = require("path");
const { Oracle } = require("./driver.js");

const RUST_CLI = path.join(__dirname, "..", "engine", "target", "release", "towerswap-cli");
let seed = +process.argv[2];
const CASE = +process.argv[3];
const ACTIONS = +(process.argv[4] || 40);
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
let cs;
for (let c = 0; c <= CASE; c++) {
  const vseed = int(1, 2147483000), mseed = int(1, 2147483000);
  let day = int(1, 58);
  if (day % 10 === 0) day += 1;
  const setup = { day, hearts: int(3, 36), achievements: Math.min(5, Math.floor((day - 1) / 10)) };
  const board = randomBoard();
  const dayoff = rnd() < 0.5;
  const actions = [];
  for (let i = 0; i < ACTIONS; i++) actions.push({ drag: [int(1, 6), int(1, 7), int(0, 3)] });
  cs = { vseed, mseed, board, setup, dayoff, actions };
}

// 원본: 첫 밤이 시작될 때까지 행동, 그 밤을 프레임 단위로 기록
const o = new Oracle();
o.newGame(cs.vseed, cs.mseed);
o.ev(`(function(s,d,h,a){ lR(s); tK=0; nq=null; nZ=null; r7=5; rg=0; tr=d; il=h; i_=h; sN=a; })`)(cs.board, cs.setup.day, cs.setup.hearts, cs.setup.achievements);
o.frame();
let k = 0;
const origLines = [];
const origP = [];
for (; k < cs.actions.length; k++) {
  const [x, y, d] = cs.actions[k].drag;
  const r = o.ev(`(function(x,y,dx,dy){ nq = dw(x,y); if(!nq) return 'notile'; nU=dx; nB=dy; oy=tK; $C(); tc = 48*(dx+dy); d9(); return 'ok'; })`)(x, y, [0, 0, -1, 1][d], [-1, 1, 0, 0][d]);
  if (r !== "ok") continue;
  // 해소 진행; 밤(상태 3)에 들어가면 기록
  let started = false, f0 = 0;
  for (let f = 0; f < 200000; f++) {
    const s = o.ev("tK");
    if (s === 3 && !started) { started = true; f0 = o.ev("ea.loops"); o.ev("window.__nid=0"); }
    if (started) {
      // 새 드래곤에 생성 순서 번호를 붙인다(목록 끝에 추가되므로 순서 = 생성 순서)
      const line = o.ev(`(function(){ var s='F'+(ea.loops-${f0}); for(var n=a4.first;n;n=n.next){var d=n.val; if(d.__id===undefined) d.__id=window.__nid++; s+=' '+d.__id+':'+d.pixelX.toFixed(6)+','+d.pixelY.toFixed(6)+','+d.health+','+(d.isDead?1:0);} return s; })()`);
      if (s !== 3) break;
      origLines.push(line);
      const pl = o.ev(`(function(){ var s='P'+(ea.loops-${f0}); for(var n=a6.first;n;n=n.next){var p=n.val; if(p.__id===undefined) p.__id=(window.__pid=(window.__pid||0))+0, window.__pid++; var tk=p.towerKind==ou?'ArrowTower':p.towerKind==oo?'Ballista':'Cannon'; s+=' '+p.__id+':'+tk+':'+p.pixelX.toFixed(4)+','+p.pixelY.toFixed(4)+':t'+(p.targetEnemy?(p.targetEnemy.__id):'-')+':e'+p.explodedTime;} return s; })()`);
      origP.push(pl);
    }
    if (s === 0 && started) break;
    if ([5, 11, 28, 108].includes(s)) break;
    o.frame();
  }
  if (started) break;
}
console.log("밤을 일으킨 행동 인덱스:", k, " 원본 기록 프레임:", origLines.length);

// Rust: 같은 행동 목록(0..k)으로 실행하며 첫 밤을 기록
const input = JSON.stringify({ ...cs, actions: cs.actions.slice(0, k + 1) });
const res = require("child_process").spawnSync(RUST_CLI, { input, env: { ...process.env, TS_TRACE: "1" }, maxBuffer: 1 << 30 });
const rustLines = res.stderr.toString().split("\n").filter((l) => l.startsWith("F"));
console.log("Rust 기록 프레임:", rustLines.length);

// 원본 기록은 tK==3인 프레임에서 frame() 전에 찍으므로, Rust의 F(n) 은 원본의 F(n+1) 시점 상태와 대응
const parse = (l) => { const [f, ...ds] = l.split(" "); return { f, ds }; };
for (let i = 0; i < Math.min(origLines.length - 1, rustLines.length); i++) {
  const a = parse(origLines[i + 1]), b = parse(rustLines[i]);
  if (a.ds.join(" ") !== b.ds.join(" ")) {
    console.log(`첫 불일치: 원본 ${a.f} / rust ${b.f}`);
    const am = new Map(a.ds.map((x) => [x.split(":")[0], x]));
    const bm = new Map(b.ds.map((x) => [x.split(":")[0], x]));
    const keys = new Set([...am.keys(), ...bm.keys()]);
    let shown = 0;
    for (const key of keys) {
      if (am.get(key) !== bm.get(key) && shown++ < 8) console.log(`  원본 ${am.get(key) || "(없음)"}\n  rust ${bm.get(key) || "(없음)"}`);
    }
    const rustP = res.stderr.toString().split("\n").filter((l) => l.startsWith("P"));
    console.log("  원본 투사체(직전):", origP[i]);
    console.log("  rust 투사체(직전):", rustP[i - 1]);
    process.exit(0);
  }
}
console.log("기록 범위에서 불일치 없음");
