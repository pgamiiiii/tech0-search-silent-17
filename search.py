import pandas as pd
import numpy as np
import re
import unicodedata
from sentence_transformers import SentenceTransformer

# =========================
# 🔹 モデル
# =========================
model = SentenceTransformer("intfloat/multilingual-e5-small")

# =========================
# 🔹 正規化
# =========================
def normalize(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").lower()

# =========================
# 🔹 Embedding作成
# =========================
def build_embedding_cache(df):
    good_texts, bad_texts = [], []

    for _, row in df.iterrows():
        good = f"{row['得意分野①']} {row['得意分野②']} {row['専門スキル']}"
        bad  = f"{row['不得意なこと①']} {row['不得意なこと②']}"

        good_texts.append(f"passage: {good}")
        bad_texts.append(f"passage: {bad}")

    return {
        "good": np.array(model.encode(good_texts)),
        "bad": np.array(model.encode(bad_texts))
    }

# =========================
# 🔥 属性フィルタ
# =========================
def apply_attribute_filters(df, query):
    filtered = df.copy()
    q_norm = normalize(query)

    # -------------------------
    # ① 名前検索（最優先）
    # -------------------------
    name_hits = df[df['氏名'].apply(normalize).str.contains(q_norm, na=False)]
    if not name_hits.empty:
        return name_hits, True

    # -------------------------
    # ② 性別
    # -------------------------
    if "男性" in query:
        filtered = filtered[filtered["性別"] == "男性"]
    elif "女性" in query:
        filtered = filtered[filtered["性別"] == "女性"]

    # -------------------------
    # ③ 部署
    # -------------------------
    for dept in df["所属部署"].dropna().unique():
        if dept in query:
            filtered = filtered[filtered["所属部署"] == dept]
            break

    # -------------------------
    # ④ 年齢（完全版）
    # -------------------------
    filtered["年齢_num"] = (
        filtered["年齢"]
        .astype(str)
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    range_match = re.search(r"(\d+)歳以上(\d+)歳未満", query)
    range_match2 = re.search(r"(\d+)[〜\-~](\d+)歳", query)
    decade_match = re.search(r"(\d+)代", query)
    age_exact_match = re.search(r"^(\d+)歳$", query.strip())
    age_match = re.search(r"(\d+)歳", query)

    if range_match:
        min_age = float(range_match.group(1))
        max_age = float(range_match.group(2))
        filtered = filtered[
            (filtered["年齢_num"] >= min_age) &
            (filtered["年齢_num"] < max_age)
        ]

    elif range_match2:
        min_age = float(range_match2.group(1))
        max_age = float(range_match2.group(2))
        filtered = filtered[
            (filtered["年齢_num"] >= min_age) &
            (filtered["年齢_num"] <= max_age)
        ]

    elif decade_match:
        base = int(decade_match.group(1))
        filtered = filtered[
            (filtered["年齢_num"] >= base) &
            (filtered["年齢_num"] < base + 10)
        ]

    elif age_exact_match:
        age = float(age_exact_match.group(1))
        filtered = filtered[
            filtered["年齢_num"] == age
        ]

    elif age_match:
        filtered = filtered[
            filtered["年齢_num"] >= float(age_match.group(1))
        ]

    # -------------------------
    # ⑤ 経験年数
    # -------------------------
    filtered["経験年数_num"] = (
        filtered["経験年数"]
        .astype(str)
        .str.extract(r'(\d+)')[0]
        .astype(float)
    )

    exp_match = re.search(r"(\d+)年", query)
    if exp_match:
        filtered = filtered[
            filtered["経験年数_num"] >= float(exp_match.group(1))
        ]

    return filtered, False

# =========================
# 🔥 スマート検索
# =========================
def smart_search(df, query, embeddings_dict, top_k=50):
    if not query.strip():
        return df.head(top_k)

    filtered_df, is_name_match = apply_attribute_filters(df, query)

    if filtered_df.empty:
        return pd.DataFrame()

    # 名前検索
    if is_name_match:
        results = filtered_df.copy()
        results["score"] = 1.0

    else:
        query_vec = model.encode([f"query: {query}"])[0]

        good_subset = embeddings_dict["good"][filtered_df.index]
        bad_subset  = embeddings_dict["bad"][filtered_df.index]

        sim_good = np.dot(good_subset, query_vec) / (
            np.linalg.norm(good_subset, axis=1) * np.linalg.norm(query_vec)
        )

        sim_bad = np.dot(bad_subset, query_vec) / (
            np.linalg.norm(bad_subset, axis=1) * np.linalg.norm(query_vec)
        )

        is_neg = any(k in query for k in ["苦手", "不得意", "低い", "ダメ", "できない"])

        combined = (sim_bad * 0.8 + sim_good * 0.2) if is_neg else (sim_good * 0.8 + sim_bad * 0.2)

        results = filtered_df.copy()
        results["score"] = np.power(combined, 3)

        if len(results) > 1:
            s_min, s_max = results["score"].min(), results["score"].max()
            if s_max - s_min > 0:
                results["score"] = (results["score"] - s_min) / (s_max - s_min)
            else:
                results["score"] = 1.0
        else:
            results["score"] = 1.0

    results = results.sort_values("score", ascending=False)
    results.insert(0, "ランク", range(1, len(results) + 1))

    return results.head(top_k)

# =========================
# 🔹 表示用
# =========================
def format_employee_card(row):
    def stars(n):
        try:
            n = int(n)
        except:
            n = 0
        return "★" * n + "☆" * (5 - n)

    return {
        "rank": row.get("ランク", "-"),
        "name": row["氏名"],
        "dept": row["所属部署"],
        "gender": row["性別"],
        "age": row["年齢"],
        "exp": f"{row['経験年数']}",
        "score": f"{row.get('score', 0):.2f}",
        "strengths": f"{row['得意分野①']} / {row['得意分野②']}",
        "weaknesses": f"{row['不得意なこと①']} / {row['不得意なこと②']}",
        "skill": stars(row["専門スキル_数値"]),
        "comm": stars(row["コミュニケーション力_数値"]),
        "leader": stars(row["リーダーシップ_数値"]),
    }