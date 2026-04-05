"""
app.py — スキルマップアプリ（search_tfidf.py 採用版）
- データ: Supabase（data_loader.load_employee_data_from_supabase）
- 検索: TF-IDF（search_tfidf.py）
- 可視化: visualizer.py（plotly）
- 要約: gpt_summarizer.py（OpenAI API）
"""

import pandas as pd
import streamlit as st

from data_loader import load_employee_data_from_supabase
from search_tfidf import (
    build_corpus,
    build_tfidf_index,
    search_employees_tfidf,
    format_employee_card,
)
from visualizer import plot_heatmap, plot_radar_chart, plot_bubble, plot_dept_bar


st.set_page_config(
    page_title="スキルマップアプリ",
    page_icon="🗺️",
    layout="wide",
)

# ── データ読み込み（キャッシュ付き）────────────────────────────
@st.cache_data(ttl=3600)        # 重たいので1回計算したらキャッシュして保持しとく
def load_data():
    # Supabaseから取得してdf化（列名は data_loader 側で日本語化される）[3]
    return load_employee_data_from_supabase()

df = load_data()


# ── TF-IDF インデックス（キャッシュ）───────────────────────────
@st.cache_resource(ttl=3600)        # ここも重たいので1回計算したらキャッシュして保持しとく
def get_tfidf_index(df_for_index: pd.DataFrame):
    """
    TF-IDF検索用インデックスを作ってキャッシュ。
    戻り値: (vectorizer, tfidf_matrix)
    """
    corpus = build_corpus(df_for_index)
    vectorizer, tfidf_matrix = build_tfidf_index(corpus)
    return vectorizer, tfidf_matrix


# ── ヘッダー ──────────────────────────────────────────────────
st.title("🗺️ スキルマップアプリ")
st.caption("従業員のスキルを検索・可視化するツール")

# ── サイドバー ────────────────────────────────────────────────
with st.sidebar:
    # 重要：キャッシュも消す（load_data のキャッシュが残ると「変わらない」現象が起きる）[2]
    if st.button("🔄 キャッシュをリセット（データ/検索）"):
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.header("🔧 フィルター")
    departments = ["全部署"] + sorted(df["所属部署"].dropna().unique().tolist())
    selected_dept = st.selectbox("部署を選択", departments)

    st.divider()
    st.header("🔍 検索設定")
    top_k = st.slider("表示する上位件数", 10, 200, 50, 10)

    st.divider()
    st.header("📊 全社統計")
    filtered = df if selected_dept == "全部署" else df[df["所属部署"] == selected_dept]

    st.metric("対象人数", f"{len(filtered)} 名")

    # 経験年数は「nn年」想定[2]
    avg_exp = filtered["経験年数"].astype(str).str.extract(r"(\d+)").astype(float).mean().iloc[0]
    st.metric("平均経験年数", f"{avg_exp:.1f} 年")

    st.metric("専門スキル平均", f"{filtered['専門スキル_数値'].mean():.1f} / 5")


# ── タブ ──────────────────────────────────────────────────────
tab_search, tab_map, tab_knowledge = st.tabs([
    "👤 従業員検索",
    "📊 スキルマップ",
    "🌐 ナレッジ統合"   # ★追加
])

# ── 従業員検索タブ ─────────────────────────────────────────────
with tab_search:
    st.subheader("キーワードで従業員を検索")

    col_input, col_opt = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "🔍 キーワード入力",
            placeholder="例: 労務管理、バックエンド開発、山田",
            label_visibility="collapsed",
        )
    with col_opt:
        show_radar = st.checkbox("レーダーチャートを表示", value=True)

    # TF-IDF検索実行
    vectorizer, tfidf_matrix = get_tfidf_index(df)

    results = search_employees_tfidf(
        df=df,
        query=query,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        department=selected_dept,   # 部署フィルタは検索関数内で処理
        top_n=top_k,                # UI負荷回避のため表示上限をここで適用
        score_threshold=0.0,
    )

    st.markdown(f"**📊 検索結果：{len(results)} 件（表示上限: {top_k}）**")
    st.divider()

    if results.empty:
        st.info("該当する従業員が見つかりませんでした。")
    else:
        for _, row in results.iterrows():
            card = format_employee_card(row)

            row_id = row.name
            with st.expander(
                f"第{card['rank']}位：{card['name']}（{card['dept']} / {card['age']} / 経験{card['exp']}） [適合度: {card['score']}]"
            ):
                col1, col2 = st.columns([2, 1]) # if show_radar else [1, 0])は、チェックボックス外すとcol2が0になってエラーを起こすからNG。

                with col1:
                    st.markdown(f"**得意分野：** {card['strengths']}")
                    st.markdown(f"**不得意：** {card['weaknesses']}")
                    st.divider()
                    st.markdown(f"専門スキル　　：{card['skill']}")
                    st.markdown(f"コミュニケーション力：{card['comm']}")
                    st.markdown(f"リーダーシップ：{card['leader']}")

                    if st.button("💬 AIスキル要約", key=f"gpt_{row_id}"):
                        with st.spinner("AIが要約中..."):
                            from gpt_summarizer import summarize_skill_with_gpt
                            summary = summarize_skill_with_gpt(row)
                        st.info(summary)

                if show_radar:
                    with col2:
                        st.plotly_chart(
                            plot_radar_chart(row),
                            use_container_width=True,
                            key=f"radar_{row_id}",
                        )

# ── スキルマップタブ ───────────────────────────────────────────
with tab_map:
    st.subheader("📊 スキルマップ")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_heatmap(df, selected_dept), use_container_width=True)
    with col2:
        filtered = df if selected_dept == "全部署" else df[df["所属部署"] == selected_dept]
        st.plotly_chart(plot_dept_bar(filtered), use_container_width=True)

    st.divider()
    st.plotly_chart(plot_bubble(df, selected_dept), use_container_width=True)

st.divider()
st.caption("© 2026 スキルマップアプリ | Powered by Streamlit + plotly")
# ── ナレッジ統合タブ ───────────────────────────
with tab_knowledge:
    st.subheader("🌐 Purpose＋Right person")

    from gpt_summarizer import summarize_text

    # =========================
    # 🧠 キャッシュ（API節約）
    # =========================
    @st.cache_data(ttl=86400)
    def cached_summary(prompt: str):
        return summarize_text(prompt)

    # =========================
    # 🔍 クエリをAIで最適化
    # =========================
    def refine_query(query):
        return cached_summary(f"""
以下の検索ワードを、社員検索に適した具体的なスキル・業務キーワードに変換してください。

検索ワード：
{query}

出力：
・スキル
・業務内容
・役割
をカンマ区切りで
""")

    # =========================
    # 🔍 入力UI
    # =========================
    query = st.text_input("例：インターナルブランディング推進チーム")

    # =========================
    # 🚀 実行
    # =========================
    if st.button("統合検索"):

        if not query.strip():
            st.warning("キーワードを入力してください")
            st.stop()

        # =========================
        # ① 意図解析
        # =========================
        with st.spinner("① 意図を解析中..."):
            refined_query = refine_query(query)

        st.info(f"🔍 解釈された検索：{refined_query}")

        # =========================
        # ② 概要生成
        # =========================
        with st.spinner("② 概要を生成中..."):
            summary = cached_summary(f"""
あなたは戦略コンサルタントです。

「{query}」について以下の形式で簡潔に説明してください。

【概要】
【重要ポイント】
・
・
・
【活用例】
""")

        st.markdown("### 🌍 概要")
        st.success(summary)

        # =========================
        # ③ 社員検索
        # =========================
        with st.spinner("③ 最適な社員を検索中..."):
            vectorizer, tfidf_matrix = get_tfidf_index(df)

            results = search_employees_tfidf(
                df=df,
                query=refined_query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                department="全部署",
                top_n=5,
                score_threshold=0.0,
            )

        # =========================
        # ④ 表示
        # =========================
        st.markdown("### 👥 推薦メンバー5人")

        if results.empty:
            st.info("該当する社員が見つかりませんでした")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), start=1):

                st.write(f"{rank}位：{row['氏名']}（{row['所属部署']}）")

                strengths = f"{row.get('得意分野①','')} / {row.get('得意分野②','')}"
                st.markdown(f"⭐ **強み**：{strengths}")

                # ✅ API使わない簡易理由（安定版）
                st.markdown(
                    f"👉 **推薦理由（簡易）**  \n"
                    f"・得意分野：{strengths}  \n"
                    
                )

                st.divider()