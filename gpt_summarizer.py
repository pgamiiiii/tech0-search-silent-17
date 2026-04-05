"""
gpt_summarizer.py — スキルマップアプリ
ChatGPT APIを使って従業員のスキルを要約する。

使い方：
    1. pip install openai
    2. 環境変数 OPENAI_API_KEY にAPIキーを設定する
    3. app.py の該当コメントアウトを外す
"""

import os

from dotenv import load_dotenv  #追加　ken
load_dotenv() #追加　ken

def build_skill_prompt(row) -> str:
    """1人分の従業員データからChatGPT用プロンプトを作成する。"""
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

要約（100文字程度）:"""


def summarize_skill_with_gpt(row) -> str:
    """
    ChatGPT APIを使って従業員のスキルを要約する。

    Args:
        row: 従業員DataFrameの1行

    Returns:
        AIによるスキル要約文字列
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは人事部門のアシスタントです。"},
                {"role": "user",   "content": build_skill_prompt(row)},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        return "（openaiライブラリが未インストールです: pip install openai）"
    except Exception as e:
        return f"（要約取得エラー: {e}）"
# =========================
# 🔥 これを一番下に追加
# =========================
def summarize_text(text: str) -> str:
    """
    任意のテキストを要約・回答生成する（RAG用）
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
