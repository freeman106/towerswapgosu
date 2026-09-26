// 원본 게임을 행동 단위로 조작하고 상태를 읽는 드라이버.
// 좌표는 명세와 같다: x=1..6(열), y=1..7(행, 1이 성 쪽). 방향: 0=상,1=하,2=좌,3=우.
"use strict";
const { loadGame } = require("./load.js");

const DIRS = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

class Oracle {
  constructor() {
    this.g = loadGame();
    this.g.onload();
    this.ev = this.g.__ev;
    // 튜토리얼·안내 팝업 끄기 (로컬 저장소 플래그)
    const ls = this.g.localStorage;
    ls["tb_0_m_leveltutorialdid1"] = "1";
    for (const id of this.ev("(function(){var r=[];for(var n=ab.first;n;n=n.next)r.push(n.val.id);return r})()")) {
      ls["tb_0_m_tilekindtutorialdid" + id] = "1";
    }
    ls["tb_0_m_safemodechest"] = "0";
  }

  // 한 프레임: 원본 _loopgo 본문과 같은 순서(로직 → update → 그리기 함수)
  frame(n = 1) {
    this.ev(`(function(n){for(var i=0;i<n;i++){ea._updatefunc();ea.update();ea._indraw=1;ea._drawfunc();ea.draw();ea._indraw=0;}})`)(n);
  }

  newGame(vseed, mseed) {
    this.ev(`(function(vs,ms){
      _C(sz);
      var ov=V.seedSet, om=M.seedSet;
      V.seedSet=function(e){ if(e!==undefined) ov.call(this,e) };
      M.seedSet=function(e){ if(e!==undefined) om.call(this,e) };
      V.seed=vs; M.seed=ms;
      gz();
      V.seedSet=ov; M.seedSet=om;
      ot=9;           // 버리기 확인창 끄기
      _X(!0);
    })`)(vseed >>> 0, mseed >>> 0);
    return this.runUntilInput();
  }

  get tK() {
    return this.ev("tK");
  }

  // 입력을 기다리는 상태가 될 때까지 프레임을 돌린다. 결정이 필요한 상태면 그 상태를 반환.
  runUntilInput(maxFrames = 200000) {
    for (let i = 0; i < maxFrames; i++) {
      const s = this.ev("tK");
      if (s === 25 && this.stopAtDusk) return 25;
      if (s === 0 && this.ev("!nq")) {
        // 한 프레임 더 돌려도 0이면 대기 상태로 본다
        this.frame();
        if (this.ev("tK") === 0) return 0;
        continue;
      }
      if (s === 13) { this.ev("mx()"); continue; }        // 보스 경고 → 확인
      if (s === 57) {                                      // 아이템 공개 창 → 클릭하면 상태 2
        if (this.ev("tO") >= 1) this.ev("iy = null, _a(2), _i(), nN = ea.loops");
        else this.frame();
        continue;
      }
      if (s === 15) { this.ev("_a(0), mx()"); continue; } // 업적 팝업 → 계속 ⚠️ 확인 필요
      if (s === 5 || s === 11 || s === 28) return s;      // 게임 오버 계열
      if (s === 108) return s;                             // 휴식일 제안
      // 플레이어 선택을 기다리는 창: 악마(56), 요정(109), 상인(83), 상점(79), 지니(72), 배치(43), TNT(116)
      if ([56, 109, 83, 79, 72, 43, 116].includes(s)) return s;
      this.frame();
    }
    throw new Error("runUntilInput: 시간 초과, tK=" + this.ev("tK"));
  }

  drag(x, y, dir) {
    const [dx, dy] = DIRS[dir];
    const r = this.ev(`(function(x,y,dx,dy){
      nq = dw(x,y); if(!nq) return 'notile';
      nU=dx; nB=dy; oy=tK; $C(); tc = 48*(dx+dy);
      var before=r7; d9();
      if (tK==121) { /* 모루 확인창: Upgrade 버튼 동작 그대로 */
        var t=dX(); t&&nq&&(dP(), ui(nq.kind,nq.tierIndex,t.gridX,t.gridY)?(nq.removeFromGrid(),nq.gridX=t.gridX,nq.gridY=t.gridY,nq.movedTime=ea.loops,ro[t.gridX][t.gridY]=nq):t.removeFromGrid(),t._delete2(),nq.pixelPositionReset(),nq.upgrade(),dV()), _a(1), nq=null;
      }
      return 'ok';
    })`)(x, y, dx, dy);
    if (r !== "ok") return r;
    return this.runUntilInput();
  }

  // 선택 창이 선택 단계(tO=n)가 될 때까지: 애니메이션은 프레임 진행, 안내 문구(tO>=1)는 클릭으로 넘김
  waitTO(n, max = 5000) {
    for (let i = 0; i < max && this.ev("tO") < n; i++) {
      if (this.ev("tO") >= 1) this.ev("_i(), nN = ea.loops");
      else this.frame();
    }
  }

  // 상태가 바뀔 때까지 프레임 진행 후 입력 대기까지
  runPast(state) {
    for (let i = 0; i < 20000 && this.ev("tK") === state; i++) {
      // 밤 직전 요정의 집 자동 창은 닫고 밤을 진행한다(시뮬레이터에서는 선택 가능한 무료 행동일 뿐)
      if (state === 25 && this.ev("tK") === 109) break;
      if (state === 56 && this.ev("tO") === 3) this.ev("_i(), 0 == tl && _i()");
      this.frame();
    }
    if (state === 25) {
      for (let k = 0; k < 20000; k++) {
        const t = this.ev("tK");
        if (t === 109) { this.ev("_a(tj), ey = null, ev = null, ex = null, r7 || (e6 = 1)"); continue; }
        if (t !== 25) break;
        this.frame();
      }
      this.stopAtDusk = false;
      const r = this.runUntilInputNoDusk();
      this.stopAtDusk = true;
      return r;
    }
    return this.runUntilInput();
  }

  runUntilInputNoDusk() {
    // 밤 진행 중에는 상태 25에서 멈추지 않는다
    return this.runUntilInput();
  }

  // engine/towerswap-cli/src/script.rs 의 interpret()와 같은 규칙
  interpret(r) {
    const ev = this.ev;
    const s = ev("tK");
    const cells = (y0) => { const a = []; for (let y = y0; y <= 7; y++) for (let x = 1; x <= 6; x++) a.push([x, y]); return a; };
    const pick = (n) => r % n;
    const X = 1 + (Math.floor(r / 7) % 6), Y = 1 + (Math.floor(r / 41) % 7);
    if (s === 0) {
      if (r % 100 < 75) return this.drag(X, Y, Math.floor(r / 293) % 4);
      return this.tap(X, Y);
    }
    if (s === 25) {
      if (r % 3 !== 0) return this.tap(X, Y);
      return this.runPast(25);
    }
    if (s === 108) { ev(r % 2 === 0 ? "_D(eU = tr, 'dayoffday'), _a(1)" : "_a(1)"); return this.runUntilInput(); }
    if (s === 56) {
      this.waitTO(2);
      if (r % 2 === 0) ev("_D(t_ += 1, 'dealsdone'), t$ > 0 ? (il -= t$, _i(), nN = ea.loops) : (_a(56), tO = 3)");
      else ev("tg += 1, _a(2), iy = null");
      return this.runPast(56);
    }
    if (s === 79) {
      this.waitTO(2);
      if (r % 2 === 0) ev("(function(){ for (var n = tz.first; n; n = n.next) if (n.val.cost < 0) rx = n.val; _a(79); tO = 3; })()");
      else ev("_a(2)");
      return this.runPast(79);
    }
    if (s === 72) {
      this.waitTO(2);
      ev(`(function(id){ tB = dI(id); _i(); _t(); })`)(["l", "s", "i", "g", "d"][pick(5)]);
      return this.runPast(72);
    }
    if (s === 83) {
      this.waitTO(2);
      const opts = cells(1).filter(([x, y]) => ev(`(function(x,y){ var t = dw(x,y); return !!(t && t.merchantBuyerCanBuy()); })`)(x, y));
      const [x, y] = opts[pick(opts.length)];
      ev(`(function(x,y){ rv = dw(x,y); _i(); gQ(rv.pixelX + 24, rv.pixelY + 24, rv.merchantAndMatch4SwapsGet(), 0); _i(); nN = ea.loops; })`)(x, y);
      return this.runPast(83);
    }
    if (s === 43) {
      const opts = cells(0).filter(([x, y]) => ev(`(function(x,y){ return !!ui(rx.tileKind, rx.tier, x, y); })`)(x, y));
      const [x, y] = opts[pick(opts.length)];
      ev(`(function(x,y){ _9(rx.tileKind, rx.tier, x, y) && (iy = null, _a(2)); })`)(x, y);
      return this.runUntilInput();
    }
    if (s === 116) {
      const opts = cells(1).filter(([x, y]) => ev(`(function(x,y){ var t = dw(x,y); return !!(t && t.tierIndex < 1 && t != oc); })`)(x, y));
      const i = pick(opts.length + 1);
      if (i === 0) ev("_a(58)");
      else ev(`(function(x,y){ o2 = dw(x,y); _a(58); oc && oc.dynamiteTotal--; })`)(...opts[i - 1]);
      return this.runUntilInput();
    }
    if (s === 109) {
      if (ev("tO") === 0) {
        const opts = cells(1).filter(([x, y]) => ev(`(function(x,y){ var t = dw(x,y); return !!(t && t.kind != e7); })`)(x, y));
        const [x, y] = opts[pick(opts.length)];
        ev(`(function(x,y){ ey = dw(x,y); _i(); })`)(x, y);
        return 109;
      }
      const opts = cells(0).filter(([x, y]) => ev(`(function(x,y){ var t = y >= 1 ? dw(x,y) : null; return !!(t ? gq(ey, t) : ui(ey.kind, ey.tierIndex, x, y)); })`)(x, y));
      if (!opts.length) { ev("_a(tj), ey = null, ev = null, ex = null, r7 || (e6 = 1)"); return this.runUntilInput(); }
      const [x, y] = opts[pick(opts.length)];
      ev(`(function(x,y){ ev = y >= 1 ? dw(x,y) : null; e4 = x; ew = y; _i(); })`)(x, y);
      return this.runPast(109);
    }
    return s;
  }

  tap(x, y) {
    const before = this.ev("tK");
    this.ev(`(function(x,y){ var t=dw(x,y); if(t){ oy=tK; da(t);} })`)(x, y);
    if (this.ev("tK") === 63) this.ev(`_a(${before})`); // 모루 정보 창(UI 전용)은 닫는다
    return this.runUntilInput();
  }

  state() {
    return JSON.parse(
      this.ev(`JSON.stringify((function(){
        var b=[];
        for(var y=1;y<=rl;y++){ var row=[]; for(var x=1;x<=6;x++){ var t=ro[x][y];
          if(!t){ row.push(dC(x,y)?'..':'~~'); continue; }
          var v=t.kind.id + (t.kind==o0? t.dynamiteTotal : t.tierIndex);
          if(t.kind==or) v+= t.flippedPreferred?'<':'>';
          row.push(v);
        } b.push(row); }
        var tur=[]; for(var x=1;x<=6;x++){ var t=iu[x]; tur.push(t? t.kind.id+t.tierIndex : ''); }
        return {tK:tK, day:tr, hearts:il, swaps:r7, sN:sN, bossCol:rw, dayOff:eU, deals:t_, declined:tg,
                vseed:V.seed>>>0, mseed:M.seed>>>0, loops:ea.loops, board:b, turrets:tur};
      })())`)
    );
  }
}

function printState(s) {
  console.log(`day=${s.day} hearts=${s.hearts} swaps=${s.swaps} tK=${s.tK} bossCol=${s.bossCol} turrets=[${s.turrets.join(",")}]`);
  for (const row of s.board) console.log("  " + row.map((c) => c.padEnd(3)).join(" "));
}

module.exports = { Oracle, printState, DIRS };

if (require.main === module) {
  const o = new Oracle();
  o.newGame(12345, 67890);
  printState(o.state());
}
