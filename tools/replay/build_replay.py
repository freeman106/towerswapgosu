"""리플레이 페이지 만들기: 짝 기록 JSON(rl/record_search_games.py 출력)을 원본 게임 그림으로 보여 주는 HTML로 만든다.
원본 텍스처는 저장소에 넣지 않는다. 없으면 게임 서버에서 reference/towerswap_v120/image/texture.png로 받는다(커밋 제외 경로).
스프라이트 좌표(sprite_rects.json)는 원본 ea.imageaddgl 목록(game.pretty.js 2047행, `1e3` 표기 주의)과 종류별 levelImages에서 얻었다.
macOS의 sips로 자른다. 결과 HTML은 게임 그림을 담으므로 비공개로만 쓴다.
사용: python tools/replay/build_replay.py pairs.json out.html"""
import base64
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEXTURE = os.path.join(ROOT, "reference", "towerswap_v120", "image", "texture.png")
URL = "https://tower-swap.game-files.crazygames.com/tower-swap/108/image/texture.png?2v=120"
SAFE = {"C": "slot", "T": "tnt", "I": "iceberg", "F": "fairy", "H": "fhouse"}  # 대소문자 구분 없는 파일 시스템 대비


def sprites():
    if not os.path.exists(TEXTURE):
        os.makedirs(os.path.dirname(TEXTURE), exist_ok=True)
        subprocess.run(["curl", "-sf", "-o", TEXTURE, URL], check=True)  # 파이썬 기본 요청은 서버가 403으로 막는다
    rects = json.load(open(os.path.join(HERE, "sprite_rects.json")))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, lst in rects.items():
            for t, (x, y, w, h) in enumerate(lst, start=1):
                f = os.path.join(tmp, f"{SAFE.get(k, k)}{t}.png")
                subprocess.run(["sips", "-c", str(h), str(w), "--cropOffset", str(y), str(x), TEXTURE, "--out", f], capture_output=True, check=True)
                out[f"{k}{t}"] = (w, h, "data:image/png;base64," + base64.b64encode(open(f, "rb").read()).decode())
    return out


def main():
    data_path, out_path = sys.argv[1], sys.argv[2]
    spr = sprites()
    css = ":root { --grass: url(%s); }\n" % spr["grass41"][2]
    css += "\n".join(".spr-%s { background-image: url(%s); }" % (k, v[2]) for k, v in spr.items() if k != "grass41")
    sizes = {k: [v[0], v[1]] for k, v in spr.items() if k != "grass41"}
    data = open(data_path).read()
    assert "</script" not in data
    t = open(os.path.join(HERE, "replay_template.html")).read()
    open(out_path, "w").write(t.replace("__SPRITE_CSS__", css).replace("__SPR_SIZES__", json.dumps(sizes)).replace("__DATA__", data))
    print(out_path)


if __name__ == "__main__":
    main()
