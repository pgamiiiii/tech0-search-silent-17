import pandas as pd
import streamlit as st
from supabase import create_client, Client

# スキルテキストを数値に変換する辞書
SKILL_LEVEL_MAP = {
    "基礎知識はあるが実務経験は浅く、サポートが必要な段階": 1,
    "定型業務は対応可能だが、複雑な場面では指導を要する": 2,
    "独力で標準的な業務を遂行でき、チーム内でも安定した貢献ができる": 3,
    "高い専門性を持ち、複雑な課題にも自律的に対応できる": 4,
    "エキスパートレベルで、組織内外で頼られる存在であり後進の指導もできる": 5,
    "意思疎通に課題があり、情報共有や連携に改善の余地がある": 1,
    "基本的なやり取りはできるが、複雑な調整や交渉は苦手": 2,
    "社内外との円滑なコミュニケーションが取れ、チームワークに貢献している": 3,
    "相手に合わせた的確な伝達力があり、交渉や折衝にも優れる": 4,
    "卓越した対話力を持ち、ステークホルダーとの関係構築や説得を得意とする": 5,
    "リーダー経験はほぼなく、現時点ではフォロワーとしての役割が中心": 1,
    "小規模なタスクのまとめ役は担えるが、チームの牽引には経験が必要": 2,
    "チームをまとめる基本的なリーダーシップを発揮でき、メンバーからの信頼も厚い": 3,
    "高いリーダーシップを持ち、多様なメンバーを束ねてプロジェクトを推進できる": 4,
    "組織全体を動かす卓越したリーダーシップを持ち、変革や難局でも力を発揮する": 5,
}

def load_employee_data_from_supabase() -> pd.DataFrame:
    """
    Supabaseの英語カラム名を日本語名に変換し、前処理済みのDataFrameを返す。
    """
    # 1. Supabaseクライアントの初期化
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)

    # 2. データの取得
    response = supabase.table("employees").select("*").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        return df

    # 3. 🔥 重要：英語の列名を日本語に変換する
    # 画像のテーブル定義（dept, name等）をapp.pyが探している日本語名に対応させます
    rename_map = {
        "name": "氏名",
        "gender": "性別",
        "age": "年齢",
        "dept": "所属部署",
        "experience_years": "経験年数",
        "strength_1": "得意分野①",
        "strength_2": "得意分野②",
        "weakness_1": "不得意なこと①",
        "weakness_2": "不得意なこと②",
        "skill_raw": "専門スキル",
        "comm_raw": "コミュニケーション力",
        "leader_raw": "リーダーシップ",
        "skill_score": "専門スキル_数値",
        "comm_score": "コミュニケーション力_数値",
        "leader_score": "リーダーシップ_数値"
    }
    df = df.rename(columns=rename_map)

    # 4. 数値が未計算（NaN or 0）の場合の補完処理
    for col, raw_col in [
        ("専門スキル_数値", "専門スキル"),
        ("コミュニケーション力_数値", "コミュニケーション力"),
        ("リーダーシップ_数値", "リーダーシップ"),
    ]:
        # 念のため raw 側の空白や不可視文字を除去（マッチ率改善）
        if raw_col in df.columns:
            df[raw_col] = df[raw_col].astype(str).str.strip()
    
        # 数値列がない場合は新規作成（ "_数値" の列に再計算結果を上書きする。列は重複しない。）
        if col not in df.columns:
            df[col] = df[raw_col].map(SKILL_LEVEL_MAP).fillna(0).astype(int)
            continue
    
        # ★ 0 も「未計算」とみなす（マップの数値が全部０になっちゃう対策）
        df[col] = pd.to_numeric(df[col], errors="coerce")
        needs_fill = df[col].isnull() | (df[col] == 0)
    
        if needs_fill.any():
            mapped = df.loc[needs_fill, raw_col].map(SKILL_LEVEL_MAP)
            df.loc[needs_fill, col] = mapped

    # 経験年数を表示用に加工（数値に"年"を付ける）
    if "経験年数" in df.columns:
        df["経験年数"] = df["経験年数"].astype(str).str.replace(".0", "", regex=False) + "年"

    return df