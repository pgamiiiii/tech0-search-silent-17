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
# ★新規追加ブロック
with tab_knowledge:
    st.subheader("🌐 Purpose＋Right person")

    # ★追加：外部検索用ライブラリ
    import urllib.request, urllib.parse, re

    # ★追加：GPT要約
    from gpt_summarizer import summarize_text

    # =========================
    # 🌍 外部記事取得ロジック
    # =========================
    # ★追加
    def build_search_urls(query):
        q = urllib.parse.quote(query)
        return [
            f"https://note.com/search?q={q}",
            f"https://itmedia.co.jp/search?q={q}",
            f"https://qiita.com/search?q={q}"
        ]

    # ★追加
    def extract_links(url):
        try:
            html = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
            links = re.findall(r'https://[^\s"]+', html)
            # ★改善：Qiita / Zennに限定
            return [l for l in links if "qiita.com" in l or "zenn.dev" in l][:5]
        except:
            return []

    # ★追加
    def fetch_text(url):
        try:
            html = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
            text = re.sub('<[^<]+?>', '', html)
            return text[:2000]
        except:
            return ""

    # ★追加
    def get_articles(query):
        urls = build_search_urls(query)
        links = []
        for u in urls:
            links += extract_links(u)

        docs = []
        for link in links[:5]:
            text = fetch_text(link)
            if len(text) > 200:
                docs.append(text)
        return docs

    # =========================
    # 🔍 入力UI
    # =========================
    # ★追加
    query = st.text_input("例：インターナルブランディング推進チーム")

    # =========================
    # 🚀 実行ボタン
    # =========================
    # ★追加
    if st.button("統合検索"):
        with st.spinner("分析中..."):

            # =========================
            # 🌍 外部知見（要約）
            # =========================
            docs = get_articles(query)
            context = "\n".join(docs[:3])

            summary = summarize_text(f"""
以下は「{query}」に関する記事です。
重要なポイントのみ200文字で要約してください。

{context}
""")

            st.markdown("### 🌍 概要")
            st.success(summary)

            # =========================
            # 👥 社員推薦（★修正ポイント）
            # =========================

            # ★重要：あなたの環境はTF-IDFなのでsmart_searchは使わない
            vectorizer, tfidf_matrix = get_tfidf_index(df)

            # ★変更：search_employees_tfidfを使用
            results = search_employees_tfidf(
                df=df,
                query=query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                department="全部署",
                top_n=10,
                score_threshold=0.0,
            )

            # =========================
            # 表示
            # =========================
            if results.empty:
                st.info("該当する社員が見つかりませんでした")
            else:
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    st.write(f"{rank}位：{row['氏名']}（{row['所属部署']}）")

                    # ★追加：強み表示
                    strengths = f"{row.get('得意分野①','')} / {row.get('得意分野②','')}"
                    st.markdown(f"⭐ **強み**：{strengths}")

                    # ★追加：AI推薦理由
                    reason = summarize_text(f"""
この社員が「{query}」に適している理由を
80〜120文字で簡潔に説明してください。

氏名：{row['氏名']}
部署：{row['所属部署']}
得意分野：{strengths}
スキル：{row['専門スキル_数値']}
""")

                    st.markdown(f"👉 **理由**：{reason}")
                    st.divider()