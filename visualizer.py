"""
visualizer.py — スキルマップアプリ
plotlyを使ったグラフ描画を行う。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_heatmap(df: pd.DataFrame, department: str = "全部署"):
    """部署別スキルヒートマップを作成する。"""
    if department != "全部署":
        df = df[df["所属部署"] == department]

    dept_skill = df.groupby("所属部署")[
        ["専門スキル_数値", "コミュニケーション力_数値", "リーダーシップ_数値"]
    ].mean().round(2)
    dept_skill.columns = ["専門スキル", "コミュニケーション力", "リーダーシップ"]

    fig = go.Figure(data=go.Heatmap(
        z=dept_skill.values,
        x=dept_skill.columns.tolist(),
        y=dept_skill.index.tolist(),
        colorscale="Blues",
        zmin=1,
        zmax=5,
        text=dept_skill.values,
        texttemplate="%{text:.2f}",
    ))
    fig.update_layout(
        title="部署別スキルマップ（平均レベル 1〜5）",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_radar_chart(row):
    """1人分のスキルをレーダーチャートで表示する。"""
    categories = ["専門スキル", "コミュニケーション力", "リーダーシップ"]
    values = [
        int(row["専門スキル_数値"]),
        int(row["コミュニケーション力_数値"]),
        int(row["リーダーシップ_数値"]),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(55, 138, 221, 0.3)",
        line=dict(color="#378ADD"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title=f"{row['氏名']} のスキルチャート",
        height=300,
        margin=dict(l=40, r=40, t=50, b=20),
        showlegend=False,
    )
    return fig


def plot_bubble(df: pd.DataFrame, department: str = "全部署"):
    """得意分野別の人材分布バブルチャートを作成する。"""
    if department != "全部署":
        df = df[df["所属部署"] == department]

    strengths = pd.concat([
        df["得意分野①"],
        df["得意分野②"],
    ]).value_counts().reset_index()
    strengths.columns = ["スキル", "人数"]

    fig = px.scatter(
        strengths,
        x="スキル",
        y="人数",
        size="人数",
        color="人数",
        color_continuous_scale="Blues",
        title="スキル別人材分布",
        height=420,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=50, b=100),
    )
    return fig


def plot_dept_bar(df: pd.DataFrame):
    """部署別人数棒グラフを作成する。"""
    dept_count = df["所属部署"].value_counts().reset_index()
    dept_count.columns = ["部署", "人数"]

    fig = px.bar(
        dept_count,
        x="部署",
        y="人数",
        color="人数",
        color_continuous_scale="Blues",
        title="部署別人数",
        height=350,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
