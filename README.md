# Phantom Track Generator [DARK WEB EDITION]

複数の既存楽曲（音声ファイル）とテキストプロンプトを入力すると、参照楽曲のスタイルを反映した新しい "幻の楽曲" を生成・再生・ダウンロードできるウェブアプリです。

## 概要

- 最大20曲の音楽トラックをアップロード（最大合計100MB）
- テキストプロンプトで希望の雰囲気を指定
- 音楽ジャンルの選択と生成パラメータのカスタマイズ
- 15秒〜2分の生成時間設定
- MusicGenモデルによる楽曲生成
- ダークウェブ風のスタイリッシュなインターフェース
- 生成された音楽の再生とダウンロード

## 使い方

### Google Colabで実行する場合（推奨）

1. 以下のColabノートブックにアクセスしてください：
   [Google Colabで開く](https://colab.research.google.com/github/Ryuto1991/phantom-track/blob/main/phantom_track.ipynb)

2. ランタイムタイプを「GPU」に設定

3. ランタイムメニューから「すべてのセルを実行」を選択

4. 実行後に表示されるGradio公開URLにアクセス

### ローカル環境で実行する場合（GPU推奨）

```bash
# リポジトリをクローン
git clone https://github.com/Ryuto1991/phantom-track.git
cd phantom-track

# 依存パッケージをインストール
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/audiocraft.git

# アプリケーションを実行
python app.py
```

実行後、ターミナルに表示されるURLにアクセスしてください。

## 拡張機能

- **最大20曲対応**: 複数のリファレンス曲から音楽的特徴を抽出
- **生成時間の調整**: 15秒から最大2分まで生成時間をカスタマイズ可能
- **ジャンル指定**: 22種類以上の音楽ジャンルから選択可能
- **詳細パラメータ調整**: 温度、トップK、トップP、ガイダンス強度など
- **ダークウェブUI**: サイバーパンク風の雰囲気あるUIデザイン

## システム要件

- Python 3.8以上
- CUDA対応GPU（推奨、10GB以上のVRAM）
- または、Google Colabの無料GPUランタイム

## 注意事項

- アップロードする音源は私的利用のみを目的としてください
- 生成された楽曲の商用利用はできません（CC-BY-NC相当）
- 著作権のある楽曲を使用する場合は、各権利者のガイドラインに従ってください
- Google Colabの無料版ではメモリ制限があります。多数の曲や長時間の生成はエラーになる場合があります

## 技術スタック

- フロントエンド: Gradio 4.x
- モデル: MusicGen-medium (facebook/audiocraft)
- オーディオ処理: pydub, torchaudio
- スタイリング: カスタムCSS, テーマカスタマイズ

## ライセンス

このプロジェクトはApache-2.0ライセンスで配布されています。
MusicGenモデル（AudioCraft）も同様にApache-2.0ライセンスです。

## 作者

[Ryuto1991](https://github.com/Ryuto1991)