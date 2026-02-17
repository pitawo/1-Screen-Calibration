# カメラキャリブレーションツール

Charucoボード（ChArUco）を使用した、高精度なカメラキャリブレーションツールです。
通常レンズ・魚眼レンズ（GoPro等）の両方に対応しており、頑健なCharucoボード検出と詳細なログ・結果出力が特徴です。

**主な特徴:**
- **頑健なCharucoボード検出**: ブレ、反射、照明変化に強い複数の検出アルゴリズムを搭載。
- **統一されたログ出力**: ターミナル表示と全く同じ内容がログファイルに保存されます。
- **人間が読める結果出力**: NPZファイルに加え、テキストファイルでも詳細なキャリブレーション結果を出力します。

---

## 必要な環境

```bash
pip install opencv-contrib-python numpy
```

---

## ファイル構成

```
コードのみで実行/
├── run_calibration_no_grid.py  ← メインのキャリブレーションスクリプト
├── README.md                   ← このファイル
└── system/
    ├── __init__.py             ← パッケージ初期化
    ├── base_calibrator.py      ← 基底キャリブレータークラス
    ├── fisheye_calibrator.py   ← 魚眼レンズ用キャリブレーション
    ├── method_j.py             ← 通常レンズ用キャリブレーション
    ├── utils.py                ← ユーティリティ関数（頑健検出、ログ機能など）
    └── interactive.py          ← 対話型インターフェース
```

---

## 1. カメラキャリブレーション (`run_calibration_no_grid.py`)

Charucoボードを使用し、位置・角度・大きさのバランスが取れたフレームを自動選択して高精度なキャリブレーションを行います。

### 対話モード（引数なしで実行）

```bash
python run_calibration_no_grid.py
```

画面の案内に従って、動画ファイルの指定、レンズタイプの選択、各種パラメータの設定を行えます。
デフォルト値は Enter キーを押すだけで適用されます。

### コマンドラインモード

```bash
# 通常レンズ（デフォルト設定: rows=5, cols=7, square=0.03, marker=0.015, DICT_5X5_100）
python run_calibration_no_grid.py 動画ファイル.mp4

# 魚眼レンズ（GoPro等）
python run_calibration_no_grid.py 動画ファイル.mp4 --fisheye

# Charucoボードパラメータを明示的に指定
python run_calibration_no_grid.py 動画ファイル.mp4 --rows 5 --cols 7 --square_size 0.03 --marker_size 0.015 --dictionary DICT_5X5_100
```

### オプション一覧

| オプション | 短縮形 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--output_dir` | `-o` | `./output` | 出力ディレクトリ |
| `--rows` | - | 5 | Charucoボード行数（マス数） |
| `--cols` | - | 7 | Charucoボード列数（マス数） |
| `--square_size` | - | 0.03 | マスのサイズ [m] |
| `--marker_size` | - | 0.015 | ArUcoマーカーサイズ [m] |
| `--dictionary` | - | DICT_5X5_100 | ArUco辞書名 |
| `--target_frames` | - | 200 | 目標フレーム数（150-300推奨） |
| `--blur_threshold` | - | 120.0 | ブレ判定の厳しさ（高いほど厳しい） |
| `--no_k_center` | - | - | フレーム自動補完を無効化 |
| `--fisheye` | - | - | 魚眼レンズモード |
| `--preview` | - | - | キャリブレーション前にプレビュー表示 |

### 出力ファイル

出力ディレクトリ（デフォルト: `output/` または動画と同じフォルダの `calibration_output/`）に以下のファイルが生成されます。

1.  **キャリブレーションデータ (`.npz`)**
    *   例: `calibration_20231027_123456.npz`
    *   Python等で読み込むための数値データ。

2.  **キャリブレーションレポート (`_result.txt`)**
    *   例: `calibration_20231027_123456_result.txt`
    *   カメラ行列、歪み係数、誤差などの詳細をテキスト形式で保存。メモ帳などで確認できます。

3.  **処理ログ (`_log_....txt`)**
    *   例: `calibration_log_20231027_123456.txt`
    *   実行時のターミナル出力と完全に同じ内容のログ。

---

## 2. npzファイルの中身

キャリブレーション結果の `.npz` ファイルには以下の情報が格納されます。

### 通常レンズ・魚眼レンズ共通

| キー | 型 | 内容 |
|------|-----|------|
| `camera_matrix` | ndarray (3x3) | カメラ内部パラメータ（焦点距離 fx, fy、光学中心 cx, cy） |
| `dist_coeffs` | ndarray | レンズ歪み補正値（通常レンズ: 5個、魚眼: 4個） |
| `rms` | float | キャリブレーション誤差（小さいほど高精度） |
| `image_size` | ndarray [w, h] | 画像の解像度 |
| `board_shape` | ndarray [rows, cols] | Charucoボードのマス数 |
| `square_size` | float | Charucoボードのマスサイズ (m) |
| `marker_size` | float | ArUcoマーカーサイズ (m) |
| `aruco_dictionary` | str | 使用したArUco辞書名 |
| `calib_flags` | int | キャリブレーション時のフラグ |
| `per_view_errors` | ndarray | フレームごとの誤差一覧 |
| `calib_date` | str | キャリブレーション実行日 |
| `new_camera_matrix` | ndarray (3x3) | 歪み補正後の最適カメラパラメータ |
| `map1` | ndarray | 歪み補正マップ（X方向） |
| `map2` | ndarray | 歪み補正マップ（Y方向） |
| `method_details_json` | str (JSON) | 使用した設定・統計情報の詳細 |

### 通常レンズのみ

| キー | 型 | 内容 |
|------|-----|------|
| `roi` | ndarray [x, y, w, h] | 歪み補正後の有効領域（黒枠を除いた範囲） |

### npzファイルの読み込み方

```python
import numpy as np
import cv2

# 読み込み
data = np.load("calibration_XXXXXXXX_XXXXXX.npz")

# 主要パラメータの取り出し
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
map1 = data["map1"]
map2 = data["map2"]

# 歪み補正（map1/map2 を使う方法 - 最も簡単）
frame = cv2.imread("input.jpg")
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

# 歪み補正（camera_matrix/dist_coeffs を使う方法）
undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, data["new_camera_matrix"])
```

---

## トラブルシューティング

### Charucoボードが検出されない
- **新しい「頑健検出モード」により、以前より検出率は向上していますが、それでも検出されない場合:**
    - 照明を均一にする（強い影や白飛びを避ける）。
    - Charucoボードが画面内に完全に収まっている時間を増やす。
    - `--rows` と `--cols` の値が正しいか再確認する（**マス数**を指定します）。

### 魚眼キャリブレーションが失敗する
- Charucoボードが極端に画面の端だけに映っているフレームが多いと失敗しやすいです。
- 画面の中央付近にもCharucoボードを映すようにしてください。
