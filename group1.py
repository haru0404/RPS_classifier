import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum, auto
from typing import List


# 円形度を計算
def circularity(contour):
    # 面積
    area = cv2.contourArea(contour)
    # 周囲長
    length = cv2.arcLength(contour, True)

    # 円形度を返す
    return 4*np.pi*area/length/length


# 抽出した輪郭の最大面積の添え字を返す
def index_max_contourArea(contours):
    # 暫定最大値、添え字を決定
    max = cv2.contourArea(contours[0])
    max_num = 0
    for i, contour in enumerate(contours):
        # 輪郭の面積を求める
        area = cv2.contourArea(contour)
        if max < area :
            max = area
            max_num = i

    return max_num


# 勝敗
class Result(Enum):
    WIN = auto()
    LOSE = auto()
    DRAW = auto()

# グー、チョキ、パーの手
class Hand(Enum):
    G = 18
    C = 4
    P = 9

# 勝敗判定（playersの添え字に対応する勝敗をresultsで返す）
def determine_winner(players: List[Hand]):
    # 場に出されている手を集める
    hands = 0
    for player in players:
        hands |= player.value
    # 勝利手の決定
    win_hand = hands & (2 + 4 + 8) & (hands >> 1) & ~(hands << 1)
    # 結果
    results = []
    for player in players:
        if win_hand == 0:
            results.append(Result.DRAW)
        elif (win_hand & player.value) != 0: #勝敗判定
            results.append(Result.WIN)
        else:
            results.append(Result.LOSE)
    return results


# 画像読み込み~~~~~~~~~~~~~~~
# コマンドライン引数用
#img = sys.argv[1]
#src = cv2.imread(img, cv2.IMREAD_COLOR)
#if src is None:
#   print("can't image open", file=sys.stderr)
#   sys.exit(1)

# 読み込み
src = cv2.imread("./sample1.png", cv2.IMREAD_COLOR)


# 0 -- 255 の uint8 型の画素値を，0 -- 1 の float32 型に変換し，
# BGR --> HSV 変換を施す
rgb = src.astype(np.float32) / 255.0
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
tmp1 = cv2.inRange(hsv, (320, 0.2, 0.1), (360, 0.7, 1.0))
tmp2 = cv2.inRange(hsv, (0, 0.2, 0.1), (40, 0.7, 1.0))
mask = cv2.bitwise_or(tmp1, tmp2)

# オープニング・クロージングによるノイズ除去
open_close = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
open_close = cv2.morphologyEx(open_close, cv2.MORPH_CLOSE, None)

nlabels, labelImage, data, center = cv2.connectedComponentsWithStats(open_close)

area = data[:, 4]

players = []
hands_data = []

#forで番号取得、切り出し
for i in range(1, nlabels):
    # target_index にターゲットのラベル番号を保存
    # 面積の降順にしたとき、上から順に選択した対象の添え字
    target_index = list(np.argsort(area))[::-1][i]

    #選択対象の面積が一定以上（手と判別）
    if area[target_index] > 10000:

        # 結果の表示
        x, y, w, h, _ = data[target_index]              # ターゲットを囲む矩形情報を変数に格納
        hands_data.append(list(data[target_index,:4]))  # 座標、幅、高さまで取得（結果表示用）

        # 原画像、オープニング・クロージング画像から NumPy のスライス機能で，その領域を切り取る
        target = src[y:y+h, x:x+w]
        target_oc = open_close[y:y+h, x:x+w]

        # 一番外側の輪郭のみを取得
        contours, hierarchy = cv2.findContours(target_oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        # 輪郭内面積が最大のものの添え字を返す（スライス領域内のノイズ誤判定防止）
        target_con = index_max_contourArea(contours)
        # 円形度の計算
        cir = circularity(contours[target_con])

        # じゃんけんの手判別（playersに0から順に手の配列格納）
        if cir < 0.2:
            players.append(Hand.P)
        elif cir < 0.5:
            players.append(Hand.C)
        else:
            players.append(Hand.G)

        nhands = i  # 抽出した手の数

    # 閾値以下の要素があった場合は終了
    else:
        break

# 勝敗判定
results = determine_winner(players)

# 結果整形
for i in range(nhands):
    hand = players[i]
    result = results[i]
    hands_data[i].append(hand.name)
    if result == Result.WIN:
        hands_data[i].append("W")
    elif result == Result.LOSE:
        hands_data[i].append("L")
    else:
        hands_data[i].append("E")

# x座標順にソート
hands_data = sorted(hands_data, key = lambda x:x[0])

for i in range(nhands):
    print(*hands_data[i])
