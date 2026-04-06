"""
gpt_summarizer.py — スキルマップアプリ
ChatGPT APIを使って従業員のスキルを要約する。

使い方：
    1. pip install openai
    2. 環境変数 OPENAI_API_KEY にAPIキーを設定する
    3. app.py の該当コメントアウトを外す
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 🔐 ローカル用 .env 読み込み
# =========================
# （Cloud では .env が無いので無害）
load_dotenv()


# =========================
# 🔑 APIキー取得（ローカル + Cloud 両対応）
# =========================
def get_api_key() -> str:
    """
    Streamlit Cloud → st.secrets を最優先
    ローカル → .env または OS の環境変数
    """
    # ① Streamlit Cloud の secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # ② ローカルの環境変数（.env で読み込まれる）
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # ③ どちらも無い場合
    raise RuntimeError("OPENAI_API_KEY が設定されていません")


# =========================
# 🧠 プロンプト生成
# =========================
def build_skill_prompt(row) -> str:
    return f"""以下の従業員のスキルと特徴を、100文字程度で簡潔に要約してください。
マネージャーが人材配置の参考にするための、実用的な説明文にしてください。

【従業員情報】
- 氏名: {row['氏名']}
- 所属: {row['所属部署']}（経験{row['経験年数']}）
- 得意分野: {row['得意分野①']}、{row['得意分野②']}
- 不得意なこと: {row['不得意なこと①']}、{row['不得意なこと②']}
- 専門スキル評価: {row['専門スキル']}
- コミュニケーション力: {row['コミュニケーション力']}
- リーダーシップ: {row['リーダーシップ']}

要約（100文字程度）:
"""


# =========================
# ✨ スキル要約（従業員1名）
# =========================
def summarize_skill_with_gpt(row) -> str:
    try:
        client = OpenAI(api_key=get_api_key())

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは人事部門のアシスタントです。"},
                {"role": "user", "content": build_skill_prompt(row)},
            ],
            max_tokens=200,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"（要約取得エラー: {e}）"


# =========================
# ✨ 任意テキスト要約（RAG用）
# =========================
def summarize_text(text: str) -> str:
    try:
        client = OpenAI(api_key=get_api_key())

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは優秀なコンサルタントです。簡潔に分かりやすく回答してください。"},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
            temperature=0.5,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"（要約エラー: {e}）"