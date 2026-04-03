"""
app.py — スキルマップアプリ
"""

import streamlit as st
# 【旧版からの変更点】ローカルデータ読み込み（load_employee_data）から、Supabase用の読み込み関数に変更
from data_loader import load_employee_data_from_supabase  # 名前を変更
# 【旧版からの変更点】単純な検索（search_employees）を廃止し、ベクトル検索用の関数（build_embedding_cache, smart_search）を追加
from search import build_embedding_cache, smart_search, format_employee_card
from visualizer import plot_heatmap, plot_radar_chart, plot_bubble, plot_dept_bar

st.set_page_config(
    page_title="スキルマップアプリ",
    page_icon="🗺️",
    layout="wide",
)

# ── データ読み込み（キャッシュ付き）────────────────────────────
@st.cache_data
def load_data():
   # 【旧版からの変更点】引数のExcelファイルパス("employee_dummy_data.xlsx")を削除し、Supabaseから取得するよう変更
   # 引数（ファイルパス）は不要になったので消します
    return load_employee_data_from_supabase()

df = load_data()


# 【旧版からの変更点】AI（ベクトル）検索を高速化するため、Embeddingデータ（得意・苦手）をSession Stateにキャッシュする処理を新規追加
# ★変更：Embeddingをセッションに保持（戻り値がタプル(2つ)になったことに対応）
if "embeddings" not in st.session_state:
    with st.spinner("AI準備中...（得意・苦手データを分離して解析中）"):
        # 新しいsearch.pyに合わせて(good_embs, bad_embs)のセットを保存
        st.session_state.embeddings = build_embedding_cache(df)

# ── ヘッダー ──────────────────────────────────────────────────
st.title("🗺️ スキルマップアプリ")
st.caption("従業員のスキルを検索・可視化するツール")

# ── サイドバー ────────────────────────────────────────────────
with st.sidebar:
    # 【旧版からの変更点】Embedding情報などを初期化してデータを最新化するためのリセットボタンを新規追加
    # ★追加：データが更新されない問題を解決するためのリセットボタン
    if st.button("🔄 検索キャッシュをリセット"):
        st.session_state.clear()
        st.rerun()

    st.header("🔧 フィルター")
    departments = ["全部署"] + sorted(df["所属部署"].unique().tolist())
    selected_dept = st.selectbox("部署を選択", departments)

    st.divider()
    st.header("📊 全社統計")
    filtered = df if selected_dept == "全部署" else df[df["所属部署"] == selected_dept]
    st.metric("対象人数", f"{len(filtered)} 名")
    
    # 【旧版からの変更点】旧版の `.str.replace('年','')` は想定外の文字列でエラーになるリスクがあったため、
    # 正規表現 (`extract(r'(\d+)')`) を使って安全に数値を抽出するように堅牢化
    # 経験年数の計算（エラー回避のためstr操作を調整）
    avg_exp = filtered['経験年数'].astype(str).str.extract(r'(\d+)').astype(float).mean().iloc[0]
    st.metric("平均経験年数", f"{avg_exp:.1f} 年")
    st.metric("専門スキル平均", f"{filtered['専門スキル_数値'].mean():.1f} / 5")

# ── タブ ──────────────────────────────────────────────────────
tab_search, tab_map = st.tabs(["👤 従業員検索", "📊 スキルマップ"])

# ── 従業員検索タブ ─────────────────────────────────────────────
with tab_search:
    st.subheader("キーワードで従業員を検索")

    col_input, col_opt = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "🔍 キーワード入力",
            # 【旧版からの変更点】ベクトル検索（意味検索）の強みを活かすため、プレースホルダーを自然言語の例文に変更
            placeholder="例: 営業が苦手、Pythonが得意",
            label_visibility="collapsed",
        )
    with col_opt:
        show_radar = st.checkbox("レーダーチャートを表示", value=True)

    # 【旧版からの変更点】旧版の `search_employees` から、Embeddingを利用した `smart_search` に変更
    # ★変更：AI検索（得意・苦手分離ロジックを反映）
    # st.session_state.embeddings には(得意用, 苦手用)の2つが入っています
    results = smart_search(
        df,
        query,
        st.session_state.embeddings,
        top_k=len(df)
    )

    # 【旧版からの変更点】新しいsmart_search内では部署絞り込みを行っていないため、検索後に部署フィルタを適用する処理を新規追加
    # ★変更：部署フィルタを適用
    if selected_dept != "全部署":
        results = results[results["所属部署"] == selected_dept]

    st.markdown(f"**📊 検索結果：{len(results)} 件**")
    st.divider()

    if results.empty:
        st.info("該当する従業員が見つかりませんでした。")
    else:
        for _, row in results.iterrows():
            card = format_employee_card(row)
            # 【旧版からの変更点】カードのタイトル部分（expander）に、検索順位（rank）とAIによる適合度スコア（score）を表示するように変更
            # ★変更：カード表示に「ランク（順位）」と「適合度（スコア）」を追加して分かりやすく
            with st.expander(
                f"第{card['rank']}位：{card['name']}（{card['dept']} / {card['age']} / 経験{card['exp']}） [適合度: {card['score']}]"
            ):
                col1, col2 = st.columns([2, 1] if show_radar else [1, 0])

                with col1:
                    st.markdown(f"**得意分野：** {card['strengths']}")
                    st.markdown(f"**不得意：** {card['weaknesses']}")
                    st.divider()
                    st.markdown(f"専門スキル　　：{card['skill']}")
                    st.markdown(f"コミュニケーション力：{card['comm']}")
                    st.markdown(f"リーダーシップ：{card['leader']}")

                # 【旧版からの変更点】要約ボタンのStreamlitキーが重複しないよう、row['No']ではなくDataFrameのindex(row.name)を使用するように変更
                row_id = row.name #追加

                if st.button("💬 AIスキル要約", key=f"gpt_{row_id}"): #変更
                        with st.spinner("AIが要約中..."):
                            from gpt_summarizer import summarize_skill_with_gpt
                            summary = summarize_skill_with_gpt(row)
                        st.info(summary)

                if show_radar:
                    with col2:
                        st.plotly_chart(
                            plot_radar_chart(row),
                            use_container_width=True,
                            key=f"radar_{row_id}"           # グラフが重複しないようにキー指定
                        )

# ── スキルマップタブ ───────────────────────────────────────────
with tab_map:
    st.subheader("📊 スキルマップ")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_heatmap(df, selected_dept), use_container_width=True)
    with col2:
        st.plotly_chart(plot_dept_bar(filtered), use_container_width=True)

    st.divider()
    st.plotly_chart(plot_bubble(df, selected_dept), use_container_width=True)

st.divider()
# 【旧版からの変更点】コピーライトの年号を 2025 から現在の 2026 に更新
st.caption("© 2026 スキルマップアプリ | Powered by Streamlit + plotly")