"""
app.py — スキルマップアプリ
Streamlit メインアプリ。2タブ構成（従業員検索 / スキルマップ）。
"""

import streamlit as st
from data_loader import load_employee_data
from search import search_employees, format_employee_card
from visualizer import plot_heatmap, plot_radar_chart, plot_bubble, plot_dept_bar

st.set_page_config(
    page_title="スキルマップアプリ",
    page_icon="🗺️",
    layout="wide",
)

# ── データ読み込み（キャッシュ付き）────────────────────────────
@st.cache_data
def load_data():
    return load_employee_data("employee_dummy_data.xlsx")

df = load_data()

# ── ヘッダー ──────────────────────────────────────────────────
st.title("🗺️ スキルマップアプリ")
st.caption("従業員のスキルを検索・可視化するツール")

# ── サイドバー ────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 フィルター")
    departments = ["全部署"] + sorted(df["所属部署"].unique().tolist())
    selected_dept = st.selectbox("部署を選択", departments)

    st.divider()
    st.header("📊 全社統計")
    filtered = df if selected_dept == "全部署" else df[df["所属部署"] == selected_dept]
    st.metric("対象人数", f"{len(filtered)} 名")
    st.metric("平均経験年数",
              f"{filtered['経験年数'].str.replace('年','').astype(float).mean():.1f} 年")
    st.metric("専門スキル平均",
              f"{filtered['専門スキル_数値'].mean():.1f} / 5")

# ── タブ ──────────────────────────────────────────────────────
tab_search, tab_map = st.tabs(["👤 従業員検索", "📊 スキルマップ"])

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

    results = search_employees(df, query, selected_dept)
    st.markdown(f"**📊 検索結果：{len(results)} 件**")
    st.divider()

    if results.empty:
        st.info("該当する従業員が見つかりませんでした。")
    else:
        for _, row in results.iterrows():
            card = format_employee_card(row)
            with st.expander(
                f"👤 {card['name']}（{card['dept']} / {card['age']} / 経験{card['exp']}）"
            ):
                col1, col2 = st.columns([2, 1] if show_radar else [1, 0])

                with col1:
                    st.markdown(f"**得意分野：** {card['strengths']}")
                    st.markdown(f"**不得意：** {card['weaknesses']}")
                    st.divider()
                    st.markdown(f"専門スキル　　：{card['skill']}")
                    st.markdown(f"コミュニケーション力：{card['comm']}")
                    st.markdown(f"リーダーシップ：{card['leader']}")

                    # AIスキル要約（APIキー設定後に使用）
                    if st.button("💬 AIスキル要約", key=f"gpt_{row['No']}"):
                        with st.spinner("AIが要約中..."):
                            from gpt_summarizer import summarize_skill_with_gpt
                            summary = summarize_skill_with_gpt(row)
                        st.info(summary)

                if show_radar:
                    with col2:
                        st.plotly_chart(
                            plot_radar_chart(row),
                            use_container_width=True,
                        )

# ── スキルマップタブ ───────────────────────────────────────────
with tab_map:
    st.subheader("📊 スキルマップ")

    # 上段：ヒートマップ + 人数棒グラフ
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_heatmap(df, selected_dept), use_container_width=True)
    with col2:
        st.plotly_chart(plot_dept_bar(filtered), use_container_width=True)

    st.divider()

    # 下段：バブルチャート（全幅）
    st.plotly_chart(plot_bubble(df, selected_dept), use_container_width=True)

st.divider()
st.caption("© 2025 スキルマップアプリ | Powered by Streamlit + plotly")
