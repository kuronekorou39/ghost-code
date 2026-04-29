"""Gradio GUI:発行済みトークン(画像/動画)の確認と、任意メディアからのトークン検出。

起動: uv run python app.py
ブラウザ: http://127.0.0.1:7860/
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ghost_code.detect import detect
from ghost_code.registry import TokenEntry, hamming, load_registry

ROOT = Path(__file__).resolve().parent


def _abs(p: str) -> str:
    return str(ROOT / p)


def _table_rows(entries: list[TokenEntry]) -> list[list[str]]:
    return [[e.id, e.label, e.bits, e.stego_path] for e in entries]


def refresh_registry():
    all_entries = load_registry()
    images = [e for e in all_entries if e.media_type == "image"]
    videos = [e for e in all_entries if e.media_type == "video"]

    img_gallery = [(_abs(e.stego_path), f"{e.id} / {e.label}") for e in images]
    img_table = _table_rows(images)
    vid_list = [_abs(e.stego_path) for e in videos]
    vid_table = _table_rows(videos)
    stat = f"画像トークン: **{len(images)}** 件 / 動画トークン: **{len(videos)}** 件"

    return img_gallery, img_table, vid_list, vid_table, stat


def run_detect(media_path: str | None, video_samples: int = 60) -> tuple[str, str, str]:
    if not media_path:
        return "（ファイルをアップロードしてください）", "", ""

    result = detect(media_path, video_samples=int(video_samples))

    if result.extracted_bits is None:
        extracted_line = "（復元できませんでした）"
    else:
        extracted_line = f"{result.extracted_bits}  (len={len(result.extracted_bits)})"

    header_kind = {"image": "画像", "video": "動画", "unknown": "不明"}[result.media_type]

    if result.success and result.matched_entry is not None:
        e = result.matched_entry
        extra = ""
        if result.media_type == "video":
            if result.frames_detected is not None:
                extra += f"\n\n**不可視フレーム検出**: {result.frames_detected} / {result.frames_total}"
            if result.visible_code:
                extra += f"\n\n**可視コード抽出**: `{result.visible_code}`(信頼度 {result.visible_confidence:.2%})"
            if result.consensus:
                extra += f"\n\n**コンセンサス**: {result.consensus}"
        verdict = (
            f"# ✅ 一致({header_kind})\n\n"
            f"**ユーザーID**: `{e.id}`\n\n"
            f"**ラベル**: {e.label}\n\n"
            f"**復元方式**: {result.method}\n\n"
            f"**Hamming 距離**: {result.hamming_distance} / {len(e.bits)}\n\n"
            f"**信頼度**: {result.confidence:.2%}"
            f"{extra}"
        )
    elif result.matched_entry is not None:
        e = result.matched_entry
        verdict = (
            f"# ⚠️ 近傍ヒット(閾値外)({header_kind})\n\n"
            f"**最接近**: `{e.id}` ({e.label})\n\n"
            f"**Hamming 距離**: {result.hamming_distance} / {len(e.bits)}\n\n"
            f"{result.message}"
        )
    else:
        verdict = f"# ❌ 検出失敗({header_kind})\n\n{result.message}"

    # 台帳との距離ランキング
    ranking_md = ""
    if result.extracted_bits:
        reg = [e for e in load_registry() if e.media_type == result.media_type]
        if reg:
            scored = sorted(
                [(e, hamming(result.extracted_bits, e.bits)) for e in reg],
                key=lambda x: x[1],
            )[:20]  # 上位 20 件
            lines = ["| rank | id | label | Hamming |", "|---|---|---|---|"]
            for rank, (e, d) in enumerate(scored, 1):
                marker = "👉 " if (result.matched_entry and e.id == result.matched_entry.id) else ""
                lines.append(f"| {rank} | {marker}{e.id} | {e.label} | {d} / {len(e.bits)} |")
            ranking_md = "\n".join(lines)

    return verdict, extracted_line, ranking_md


with gr.Blocks(title="ghost-code: 透かしトークン検出デモ") as demo:
    gr.Markdown("# ghost-code 透かしトークン検出デモ\n"
                "各 stego メディアには見えないビット列(ユーザーID)が埋め込まれています。\n"
                "検出タブに任意の画像 or 動画をドロップすると、どのユーザーIDのコピーかを特定します。")
    status_md = gr.Markdown()

    with gr.Tab("1. 発行済みトークン(画像)"):
        gr.Markdown("### 発行済み stego 画像一覧")
        refresh_btn_i = gr.Button("🔄 台帳を再読込")
        img_gallery = gr.Gallery(label="stego 画像", columns=5, height="auto", allow_preview=True)
        img_table = gr.Dataframe(
            headers=["id", "label", "bits", "stego_path"],
            interactive=False, wrap=True,
        )

    with gr.Tab("2. 発行済みトークン(動画)"):
        gr.Markdown("### 発行済み stego 動画一覧")
        refresh_btn_v = gr.Button("🔄 台帳を再読込")
        vid_list = gr.Files(label="stego 動画(クリックで DL / 再生)")
        vid_table = gr.Dataframe(
            headers=["id", "label", "bits", "stego_path"],
            interactive=False, wrap=True,
        )

    with gr.Tab("3. 検出(画像 or 動画)"):
        gr.Markdown("### 任意ファイルから埋め込まれたトークンを検出")
        with gr.Row():
            with gr.Column(scale=1):
                media_in = gr.File(
                    label="検査対象(画像 or 動画)",
                    file_types=["image", "video"],
                    type="filepath",
                )
                video_samples_slider = gr.Slider(
                    minimum=20, maximum=200, value=60, step=10,
                    label="動画フレームサンプル数(多いほど検出率↑・時間も増)",
                )
                detect_btn = gr.Button("🔍 検出実行", variant="primary")
            with gr.Column(scale=1):
                verdict_md = gr.Markdown(label="判定")
                extracted_text = gr.Textbox(label="復元されたビット列", lines=1, interactive=False)
                ranking_md = gr.Markdown(label="台帳との距離(上位20件)")

    refresh_btn_i.click(
        fn=refresh_registry, inputs=None,
        outputs=[img_gallery, img_table, vid_list, vid_table, status_md],
    )
    refresh_btn_v.click(
        fn=refresh_registry, inputs=None,
        outputs=[img_gallery, img_table, vid_list, vid_table, status_md],
    )
    demo.load(
        fn=refresh_registry, inputs=None,
        outputs=[img_gallery, img_table, vid_list, vid_table, status_md],
    )
    detect_btn.click(
        fn=run_detect, inputs=[media_in, video_samples_slider],
        outputs=[verdict_md, extracted_text, ranking_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
