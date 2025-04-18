import os
import sys
import tempfile
import time
import traceback
import torch
import torchaudio
import gradio as gr
from pydub import AudioSegment
from typing import List, Optional, Dict, Any, Union

# audiocraft関連のインポートを修正
try:
    from audiocraft.models import MusicGen
except ImportError:
    from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write

# デバッグ用設定
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("phantom-track")

# 環境変数の設定
os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "phantom-track")
os.makedirs(TEMP_DIR, exist_ok=True)

# グローバル変数
model = None

def load_model():
    """MusicGenモデルをロードする"""
    global model
    if model is None:
        print("MusicGenモデルを読み込み中...")
        model = MusicGen.get_pretrained("facebook/musicgen-medium")
        model.set_generation_params(duration=30)  # 30秒の楽曲を生成
        print("モデル読み込み完了！")
    return model

def blend_tracks(files: List[str], duration: int = 10, crossfade_duration: int = 2) -> str:
    """複数の音声トラックを混合し、一時ファイルとして保存する
    
    Args:
        files: 音声ファイルのパスリスト
        duration: 各トラックから抽出する秒数
        crossfade_duration: クロスフェードの秒数
        
    Returns:
        混合した音声ファイルの一時パス
    """
    if not files:
        raise ValueError("音声ファイルが選択されていません")
    
    # 最大ファイル数をチェック
    if len(files) > 20:
        files = files[:20]  # 最初の20曲のみ使用
    
    # 各トラックから指定秒数を抽出し混合
    segments = []
    for file_path in files:
        if file_path.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            try:
                audio = AudioSegment.from_file(file_path)
                # トラックが短すぎる場合は全体を使用
                if len(audio) <= duration * 1000:
                    segments.append(audio)
                else:
                    # 曲の冒頭から指定秒数を抽出
                    segments.append(audio[:duration * 1000])
            except Exception as e:
                print(f"ファイル {file_path} の読み込みに失敗しました: {e}")
                continue
    
    # セグメントがない場合はエラー
    if not segments:
        raise ValueError("有効な音声ファイルがありません")
    
    # トラック数に応じてクロスフェード時間を調整
    if len(segments) > 5:
        # トラック数が多い場合はクロスフェード時間を短くする
        crossfade_ms = min(crossfade_duration * 1000, 1000)  # 最大1秒
    else:
        crossfade_ms = crossfade_duration * 1000
    
    # クロスフェードでセグメントを結合
    result = segments[0]
    for segment in segments[1:]:
        # クロスフェードで結合
        result = result.append(segment, crossfade=crossfade_ms)
    
    # 一時ファイルに保存
    output_path = os.path.join(TEMP_DIR, f"blended_{int(time.time())}.wav")
    result.export(output_path, format="wav")
    return output_path

def generate(track_paths: List[str], prompt: str, duration: int = 30, 
           genre: str = None, temperature: float = 1.0, top_k: int = 250, top_p: float = 0.0,
           classifier_free_guidance: float = 3.0) -> str:
    """参照トラックとプロンプトに基づいて新しい楽曲を生成する
    
    Args:
        track_paths: 参照音声ファイルのパスリスト
        prompt: 生成に使用するテキストプロンプト
        duration: 生成する楽曲の長さ（秒）
        genre: 音楽ジャンル（指定する場合はプロンプトに追加）
        temperature: 生成の多様性（高いほど創造的だが一貫性が低下）
        top_k: 考慮する次トークンの数
        top_p: 確率質量で考慮するトークンの割合（0.0=無効）
        classifier_free_guidance: ガイダンス強度（高いほどプロンプトに忠実）
        
    Returns:
        生成した音声ファイルのパス
    """
    # 入力値の検証
    if not track_paths or len(track_paths) == 0:
        return "音声ファイルが選択されていません。少なくとも1つのファイルをアップロードしてください。"
    
    # モデルをロード
    try:
        model = load_model()
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        return f"モデルのロード中にエラーが発生しました: {str(e)}"
    
    # 参照トラックを混合
    try:
        reference_track = blend_tracks(track_paths)
    except ValueError as e:
        return str(e)
    except Exception as e:
        print(f"トラック混合エラー: {e}")
        return f"音声ファイルの処理中にエラーが発生しました: {str(e)}"
    
    # リファレンスオーディオを読み込む
    try:
        reference_audio, sr = torchaudio.load(reference_track)
        if sr != 48000:
            reference_audio = torchaudio.transforms.Resample(sr, 48000)(reference_audio)
        
        # モノラル化と正規化
        if reference_audio.size(0) > 1:
            reference_audio = torch.mean(reference_audio, dim=0, keepdim=True)
        reference_audio = reference_audio / (torch.max(torch.abs(reference_audio)) + 1e-8)  # ゼロ除算を防止
    except Exception as e:
        print(f"音声処理エラー: {e}")
        return f"音声データの処理中にエラーが発生しました: {str(e)}"
    
    # プロンプトが空の場合はデフォルト値を設定
    if not prompt or not prompt.strip():
        prompt = "smooth melodic music"
    
    # ジャンルが指定されている場合はプロンプトに追加
    if genre and genre != "なし":
        prompt = f"{genre}, {prompt}"
    
    # 生成パラメータを設定
    try:
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=classifier_free_guidance
        )
        
        # 条件付き生成（メロディと説明）
        wav = model.generate_with_chroma([prompt], reference_audio.unsqueeze(0), progress=True)
        
        # 生成した音声を保存
        output_filename = f"phantom_track_{int(time.time())}"
        output_path = os.path.join(TEMP_DIR, output_filename)
        audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        
        return f"{output_path}.wav"
    except Exception as e:
        print(f"生成エラー: {e}")
        return f"音楽生成中にエラーが発生しました: {str(e)}"

# Gradio インターフェース
def create_ui():
    """Gradio UIを作成する"""
    
    # ジャンルの選択肢
    genre_choices = ["なし", "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip Hop", "R&B", 
                     "Country", "Folk", "Ambient", "Lo-Fi", "Trap", "Funk", "Soul", "Disco", 
                     "City Pop", "Metal", "Punk", "Blues", "Reggae", "World", "Dark Ambient", 
                     "Industrial", "Techno", "Cyberpunk", "Glitch"]
    
    # Gradio 3.50.0用にテーマを単純化
    # 3.50.0ではセットアップ方法が異なるため、シンプルなテーマを使用
    dark_theme = gr.themes.Soft()
    
    # カスタムCSS - UIのダークウェブ風スタイルを追加
    custom_css = """
    body, .gradio-container {
        background-color: #000000 !important;
        color: #00ff00 !important;
    }
    .gradio-container {
        max-width: 100% !important;
    }
    .title-box {
        background: linear-gradient(to right, #000000, #0a0a0a);
        border-left: 3px solid #8a00e6;
        border-bottom: 1px solid #222222;
        padding: 1rem;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 10px #8a00e6, 0 0 20px #b200ff;
        box-shadow: 0 3px 10px rgba(138, 0, 230, 0.2);
    }
    .title-box h1 {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        letter-spacing: 2px;
        color: #00ff00 !important;
        margin-bottom: 0.5rem;
    }
    .title-box h2 {
        font-family: 'Courier New', monospace;
        color: #00cc00 !important;
        font-size: 1.2rem;
        font-weight: normal;
    }
    .footer {
        text-align: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #222222;
        color: #666666;
        font-size: 0.8rem;
    }
    .status-generating {
        color: #ff9900 !important;
        text-shadow: 0 0 5px #ff9900;
        animation: pulse 1.5s infinite;
    }
    .status-complete {
        color: #00ff00 !important;
        text-shadow: 0 0 5px #00ff00;
    }
    .status-waiting {
        color: #8a00e6 !important;
    }
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    .gr-button-primary {
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        background-color: #8a00e6 !important;
        color: white !important;
    }
    .gr-button-primary:hover {
        background-color: #b200ff !important;
    }
    .gr-accordion {
        border: 1px solid #222222;
        background: #0a0a0a;
    }
    
    /* ダークウェブ風UIを強化するカスタムスタイル */
    input, textarea, select, .gr-box, .gr-panel {
        background-color: #101010 !important;
        border: 1px solid #333333 !important;
        color: #00ff00 !important;
    }
    .gr-box, .gr-panel {
        background-color: #0a0a0a !important;
    }
    .gr-input-label, .gr-box-top, .gr-box-top * {
        color: #00ff00 !important;
    }
    .gr-slider-handle {
        background-color: #8a00e6 !important;
    }
    .gr-slider-track {
        background-color: #222222 !important;
    }
    .gr-slider-track.gr-slider-track-active {
        background-color: #6600cc !important;
    }
    """
    
    # UIの作成
    with gr.Blocks(title="Phantom Track Generator: Dark Web Edition", theme=dark_theme, css=custom_css) as app:
        with gr.Column():
            with gr.Box(elem_classes="title-box"):
                gr.Markdown("""
                # 🎭 PHANTOM TRACK GENERATOR [DARK WEB EDITION]
                
                ## 【 ᴄʀʏᴘᴛ⧱ᴄᴏᴅᴇ: ᴍᴜsɪᴄ ᴍᴀɴɪᴘᴜʟᴀᴛɪᴏɴ sʏsᴛᴇᴍ 】
                
                1. リファレンストラックを最大20曲まで投入 (100MB以下)
                2. 生成パラメータを調整して音源コードを最適化
                3. 「INITIATE GENERATION」を実行
                """)
                
                gr.Markdown("""
                <p style="color:#666666;font-size:0.8rem;text-align:right;">⚠ WARNING: 一部のリファレンストラックは監視されています ⚠<br>
                サーバー接続は暗号化されていません</p>
                """)
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_files = gr.File(
                    label="リファレンストラックファイル (最大20ファイル)",
                    file_types=["audio"],
                    file_count="multiple"
                )
                
                prompt = gr.Textbox(
                    label="生成指示コード", 
                    placeholder="例: ネオン街を流れる暗黒シンセウェイブ、グリッチノイズが断続的に混入",
                    info="生成システムに対する指示を入力してください"
                )
                
                genre = gr.Dropdown(
                    choices=genre_choices,
                    value="なし",
                    label="音楽カテゴリ分類",
                    info="特定のジャンルクラスタに近づけたい場合に選択"
                )
                
                with gr.Accordion("高度パラメータ設定", open=False):
                    duration = gr.Slider(
                        label="生成時間長 [秒]",
                        minimum=15,
                        maximum=120,
                        value=30,
                        step=5,
                        info="出力トラックの長さ (15~120秒)"
                    )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            label="ランダム化係数",
                            minimum=0.1,
                            maximum=1.5,
                            value=1.0,
                            step=0.05,
                            info="高値=予測不能/低値=安定性 (0.1~1.5)"
                        )
                        
                        cfg = gr.Slider(
                            label="指示忠実度",
                            minimum=1.0,
                            maximum=7.0,
                            value=3.0,
                            step=0.5,
                            info="高値=指示に忠実/低値=自由解釈 (1.0~7.0)"
                        )
                    
                    with gr.Row():
                        top_k = gr.Slider(
                            label="トークン多様性 [K]",
                            minimum=50,
                            maximum=500,
                            value=250,
                            step=10,
                            info="生成多様性パラメータ (50~500)"
                        )
                        
                        top_p = gr.Slider(
                            label="確率質量制限 [P]",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            info="生成サンプリング閾値 (0=無効)"
                        )
                
                with gr.Row():
                    generate_btn = gr.Button("INITIATE GENERATION", variant="primary")
            
            with gr.Column(scale=2):
                output_audio = gr.Audio(
                    label="生成されたトラック", 
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                
                status = gr.Markdown("### <span class='status-waiting'>システム待機中...</span>", elem_id="status-display")
        
        # イベント紐付け
        def process_generation(files, prompt, genre, duration, temperature, top_k, top_p, cfg):
            """生成処理をラップした関数"""
            try:
                # ステータス更新（これが直接は使えないため、戻り値として返す）
                status_html = f"### <span class='status-generating'>⏳ 生成プロトコル実行中... [{duration}秒]</span>"
                
                # ファイル名の抽出（Gradio 3.50.0では異なる形式でファイルが渡される可能性がある）
                track_paths = []
                if files:
                    for f in files:
                        if isinstance(f, dict) and 'name' in f:
                            track_paths.append(f['name'])
                        elif hasattr(f, 'name'):
                            track_paths.append(f.name)
                        elif isinstance(f, str):
                            track_paths.append(f)
                
                # 音楽生成
                result = generate(
                    track_paths,
                    prompt,
                    duration=duration,
                    genre=genre,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    classifier_free_guidance=cfg
                )
                
                return status_html, result
            except Exception as e:
                print(f"処理エラー: {e}")
                error_message = f"エラーが発生しました: {str(e)}"
                return f"### <span style='color:red !important'>⚠️ {error_message}</span>", None

        # ボタンクリックイベントを設定
        generate_btn.click(
            fn=process_generation,
            inputs=[audio_files, prompt, genre, duration, temperature, top_k, top_p, cfg],
            outputs=[status, output_audio]
        ).then(
            fn=lambda: "### <span class='status-complete'>✓ 生成完了: トラック利用可能</span>",
            outputs=[status]
        )
        
        # 高度情報タブ
        with gr.Accordion("使用プロトコルとガイドライン", open=False):
            gr.Markdown("""
            ### 効果的な指示コード例:
            
            - `深夜のアンダーグラウンドクラブ、低音が支配する暗黒ビート、機械的なノイズが断続的に混入`
            - `80年代のビンテージシンセサイザー、ノスタルジックな暗さを持つレトロウェイブ`
            - `地下研究施設の異常なサウンドスケープ、不安定なグリッチとダークアンビエント`
            - `サイバーパンク都市の雨の日、遠くに聞こえる工場のノイズとシンセメロディ`
            - `禁断の実験音源、聞き手を不安にさせる低周波と歪んだサンプル`
            
            ### 高度パラメータ最適化ガイド:
            
            - **ランダム化係数**: 値を上げると予測不能な結果に、下げると入力に忠実な結果に
            - **指示忠実度**: 値を上げるとテキスト指示に厳密に従い、下げると自由な解釈に
            - **トークン多様性**: 生成のバリエーションを制御
            - **確率質量制限**: 生成のランダム性と一貫性のバランスを調整
            
            ### セキュリティ警告:
            コンテンツは外部サーバーに送信され、分析される可能性があります。法的に問題のある音源の使用は避けてください。
            """)
        
        with gr.Box(elem_classes="footer"):
            gr.Markdown("""
            PHANTOM TRACK GENERATOR v2.0 [DARK WEB EDITION] © 2025 | [GitHub](https://github.com/Ryuto1991/phantom-track) | すべての操作は記録されています
            """)
        
        # 初期化時にモデルを非同期でロード
        app.load(fn=lambda: load_model(), queue=False)
        
    return app

# アプリ起動
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)  # Gradio Liveでの公開URLを生成