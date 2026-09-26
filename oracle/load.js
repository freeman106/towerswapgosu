// 원본 Tower Swap(game.js v120)을 Node에서 화면 없이 띄우는 로더.
// - 브라우저 API는 스텁으로 대체한다.
// - 네트워크 요청은 전부 차단한다(개발자 서버로 기록이 가지 않도록).
// - 클로저 내부 접근용으로 window.__ev(code)를 주입한다.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const crypto = require("crypto");

const GAME_JS = path.join(__dirname, "..", "reference", "towerswap_v120", "game.js");
const GAME_SHA256 = "31714fdbe261a9f34d9efc9459609e92240ef115c5e1251d4e0e68bc32e9fcfa";

// 어떤 속성 접근·호출·생성에도 다시 스텁을 돌려주는 객체. 숫자 문맥에서는 0이 된다.
function makeStub() {
  const store = {};
  const target = function () {};
  return new Proxy(target, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => 0;
      if (k === "toString" || k === "valueOf") return () => 0;
      if (k === "then") return undefined; // Promise로 오인되지 않게
      if (k in store) return store[k];
      if (k in t) return t[k];
      return (store[k] = makeStub());
    },
    set(t, k, v) {
      store[k] = v;
      return true;
    },
    has() {
      return true;
    },
    apply() {
      return makeStub();
    },
    construct() {
      return makeStub();
    },
  });
}

// 네트워크 차단: 요청을 보내지 않고, 응답도 오지 않는다.
class BlockedXHR {
  constructor() {
    this.readyState = 0;
    this.status = 0;
    this.responseText = "";
  }
  open() {}
  send() {}
  setRequestHeader() {}
  abort() {}
  addEventListener() {}
}

function loadGame({ quiet = true } = {}) {
  let src = fs.readFileSync(GAME_JS, "utf8");
  const sha = crypto.createHash("sha256").update(src).digest("hex");
  if (sha !== GAME_SHA256) throw new Error(`game.js 해시 불일치: ${sha}`);

  // 클로저 첫 var 선언 직후에 eval 훅 주입
  const anchor = 'ee="tb_0_m_",ea={};';
  if (src.split(anchor).length !== 2) throw new Error("주입 지점을 찾지 못함");
  src = src.replace(anchor, anchor + "window.__ev=function(__c){return eval(__c)};");

  const storage = {};
  const noop = () => {};
  const sandbox = {
    console: quiet ? { log: noop, warn: noop, error: console.error, info: noop, debug: noop } : console,
    Math, Date, JSON, Array, Object, String, Number, Boolean, RegExp, Error, TypeError,
    Map, Set, WeakMap, Symbol, Promise, Proxy, Reflect, parseInt, parseFloat, isNaN, isFinite,
    encodeURIComponent, decodeURIComponent, Uint8Array, Float32Array, Int32Array, ArrayBuffer,
    localStorage: storage,
    navigator: { userAgent: "node", platform: "MacIntel", storage: makeStub(), language: "en" },
    document: makeStub(),
    location: { protocol: "https:", host: "localhost", href: "https://localhost/", search: "" },
    Image: function () { return makeStub(); },
    Audio: function () { return makeStub(); },
    XMLHttpRequest: BlockedXHR,
    fetch: () => new Promise(noop),
    requestAnimationFrame: noop,
    setTimeout: () => 0,
    clearTimeout: noop,
    setInterval: () => 0,
    clearInterval: noop,
    addEventListener: noop,
    removeEventListener: noop,
    scrollTo: noop,
    innerWidth: 400,
    innerHeight: 800,
    devicePixelRatio: 1,
    gPlatform: "web",
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  // HTML에 있던 요소들(galldiv 등)은 전역 id로 접근되므로 스텁을 둔다
  for (const id of ["galldiv", "ghtmldiv", "ghtml2div", "gxsolladiv", "gxsolladivx", "gxsollaiframediv", "gglcanvas", "gtex", "gfginputs", "gbginput", "gMerchBannerDiv"]) {
    sandbox[id] = makeStub();
  }
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox, { filename: "game.js" });
  sandbox.window.access = sandbox.access; // access()가 즉시 실행되는 구조인지 확인용
  return sandbox;
}

module.exports = { loadGame, makeStub };

if (require.main === module) {
  const g = loadGame();
  console.log("loaded. __ev:", typeof g.__ev, " onload:", typeof g.onload);
}
