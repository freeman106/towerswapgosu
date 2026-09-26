# 원본 사본 (저장소 미포함)

원본 게임 코드는 저장소에 넣지 않는다. 원본 대조(`oracle/`)를 돌리려면 직접 받아 둔다.

```sh
mkdir -p reference/towerswap_v120
curl -o reference/towerswap_v120/game.js 'https://tower-swap.game-files.crazygames.com/tower-swap/108/game.js?116'
(cd reference/towerswap_v120 && shasum -a 256 -c game.js.sha256)
```
