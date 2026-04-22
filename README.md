# ghost-code

動画への不可視フォレンジック透かし(forensic watermarking)システム。

配布する動画のコピー1つ1つに人間の目では見えない個別トークンを埋め込み、
流出時にキャプチャ動画から犯人を特定することを目的とする。

## アーキテクチャ概要

- 埋め込み: 深層学習ベース不可視透かし(画面撮影耐性)
- 配布: A/B セグメント方式(再エンコード不要、ffmpeg concat)
- 検出: 流出動画から bit 列抽出 → Reed-Solomon 復号 → ユーザーID
- ECC: Reed-Solomon(将来 Tardos 符号で結託耐性)

## ディレクトリ構成

```
src/ghost_code/   コアロジック
tests/            ユニットテスト
notebooks/        実験・検証ノートブック
data/             入出力データ(git 管理外)
  raw/            原本動画
  processed/      埋め込み済み動画/フレーム
  captures/       スマホ撮影画像など
outputs/          実験出力(git 管理外)
models/weights/   学習済みモデル(git 管理外)
```

## 開発フェーズ

- Phase 1: 静止画への埋め込み・画面撮影耐性検証 ← 現在
- Phase 2: 動画への全フレーム埋め込み・キャプチャ動画から復元
- Phase 3: A/B セグメント + RS 符号で user_id 復元
- Phase 4: サーバー化(自宅GPU + AWS Lambda + S3)
- Phase 5: Tardos 結託耐性(オプション)

## セットアップ

```powershell
uv sync
```
