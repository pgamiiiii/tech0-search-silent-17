"""
search_tfidf.py
---------------
TF-IDF を使った従業員検索（ランキング）モジュール。

目的:
- 各従業員を「1ドキュメント」として TF-IDF 化
- クエリとのコサイン類似度でスコアリング
- スコア降順で上位N件を返す
- app.py の表示を維持するため、以下も提供:
    - 結果DataFrameに 'ランク' と 'score' を付与
    - format_employee_card(row) を提供（順位・適合度・★表示）
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# 1. コーパス（検索対象テキスト）の構築
# ──────────────────────────────────────────────
def build_corpus(df: pd.DataFrame) -> list[str]:
    """
    各従業員の情報を1つの文字列（ドキュメント）に結合してコーパスを作る。
    TF-IDF は「1つのドキュメント = 従業員1人分の全テキスト」として扱う。

    Returns:
        各従業員のテキストを並べたリスト（df の行順と一致）
    """
    search_cols = [
        "氏名",
        "所属部署",
        "得意分野①",
        "得意分野②",
        "不得意なこと①",
        "不得意なこと②",
        "専門スキル",
        "コミュニケーション力",
        "リーダーシップ",
    ]

    corpus: list[str] = []
    for _, row in df.iterrows():
        parts = []
        for col in search_cols:
            if col in df.columns and pd.notna(row.get(col)):
                parts.append(str(row.get(col)))
        corpus.append(" ".join(parts))
    return corpus


# ──────────────────────────────────────────────
# 2. TF-IDF ベクトライザの構築（インデックス作成）
# ──────────────────────────────────────────────
def build_tfidf_index(corpus: list[str]):
    """
    コーパス全体に対して TF-IDF 行列を計算し、
    ベクトライザと行列をセットで返す。

    Returns:
        (vectorizer, tfidf_matrix)
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",     # 日本語向けに文字n-gram
        ngram_range=(2, 3),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


# ──────────────────────────────────────────────
# 3. TF-IDF スコアリング検索（app.py互換：ランク/score付与）
# ──────────────────────────────────────────────
def search_employees_tfidf(
    df: pd.DataFrame,
    query: str,
    vectorizer,
    tfidf_matrix,
    department: str = "全部署",
    top_n: int = 20,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    TF-IDF コサイン類似度を使ってクエリに近い従業員を上位から返す。

    Returns:
        スコア付き・降順の DataFrame
        - similarity_score: 生スコア（0〜1）
        - score: app.py表示用スコア（similarity_scoreのコピー）
        - ランク: 1位,2位,... の順位
    """
    # 空クエリなら「スコア0」で top_n 件だけ返す（全件返しはUIが重いので禁止）
    if not query or not query.strip():
        result = df.copy()
        result["similarity_score"] = 0.0
        if department != "全部署":
            result = result[result["所属部署"] == department]
        result = result.head(top_n).reset_index(drop=True)
        result["score"] = 0.0
        result.insert(0, "ランク", range(1, len(result) + 1))
        return result

    # ① クエリをベクトル化
    query_vec = vectorizer.transform([query])

    # ② コサイン類似度を計算
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # ③ スコア付与
    result = df.copy()
    result["similarity_score"] = scores

    # ④ 部署フィルター
    if department != "全部署":
        result = result[result["所属部署"] == department]

    # ⑤ しきい値フィルター
    result = result[result["similarity_score"] >= score_threshold]

    # ⑥ ソート＆上位
    result = (
        result.sort_values("similarity_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True)
    )

    # ★ app.py互換列（順位と表示用スコア）
    result["score"] = result["similarity_score"]
    result.insert(0, "ランク", range(1, len(result) + 1))

    return result


# ──────────────────────────────────────────────
# 4. 表示用：カード整形（app.py互換）
# ──────────────────────────────────────────────
def format_employee_card(row) -> dict:
    """
    app.py の expander 表示を維持するための辞書を返す。
    必要キー: rank, name, dept, age, exp, score, strengths, weaknesses, skill, comm, leader
    """
    def stars(n):
        try:
            n = int(n)
        except Exception:
            n = 0
        return "★" * n + "☆" * (5 - n)

    # 「年」「歳」表記はdf側の形式に合わせてそのまま表示（app.py側に寄せない）
    return {
        "rank": row.get("ランク", "-"),
        "name": row.get("氏名", ""),
        "dept": row.get("所属部署", ""),
        "gender": row.get("性別", ""),
        "age": row.get("年齢", ""),
        "exp": f"{row.get('経験年数', '')}",
        "score": f"{float(row.get('score', 0.0)):.2f}",
        "strengths": f"{row.get('得意分野①', '')} / {row.get('得意分野②', '')}",
        "weaknesses": f"{row.get('不得意なこと①', '')} / {row.get('不得意なこと②', '')}",
        "skill": stars(row.get("専門スキル_数値", 0)),
        "comm": stars(row.get("コミュニケーション力_数値", 0)),
        "leader": stars(row.get("リーダーシップ_数値", 0)),
    }
