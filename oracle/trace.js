// 대조용 트레이스 생성기.
// 입력(stdin, JSON 한 줄에 한 케이스): {"vseed":u32,"mseed":u32,"actions":[{"drag":[x,y,d]}|{"tap":[x,y]}|{"dayoff":true|false}]}
// 출력(stdout, JSON 한 줄에 한 케이스): {"states":[시작 상태, 행동1 후 상태, ...]}
"use strict";
const readline = require("readline");
const { Oracle } = require("./driver.js");

function runCase(c) {
  const o = new Oracle();
  o.newGame(c.vseed, c.mseed);
  const states = [o.state()];
  for (const a of c.actions || []) {
    const s = o.ev("tK");
    if (s === 5 || s === 11 || s === 28) break; // 게임 오버
    if (a.drag) o.drag(...a.drag);
    else if (a.tap) o.tap(...a.tap);
    else if ("dayoff" in a) {
      if (a.dayoff) o.ev("_D(eU = tr, 'dayoffday'), _a(1)");
      else o.ev("_a(1)");
      o.runUntilInput();
    } else throw new Error("unknown action " + JSON.stringify(a));
    states.push(o.state());
  }
  return { states };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  if (!line.trim()) return;
  process.stdout.write(JSON.stringify(runCase(JSON.parse(line))) + "\n");
});
