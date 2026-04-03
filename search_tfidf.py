"""
ranking.py
---------------
Level 2 発展課題：TF-IDF を使った従業員検索スコアリング

Week4 の ranking.py の考え方をスキルマップアプリに応用したモジュール。
通常のキーワード一致検索に代わり、各従業員との関連度スコアを計算して
上位から順に結果を返す。
"""

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
    検索対象にしたいカラムをすべてスペース区切りで連結する。

    Args:
        df: 従業員データの DataFrame

    Returns:
        各従業員のテキストを並べたリスト（df の行順と一致）
    """
    search_cols = [
        '氏名',
        '所属部署',
        '得意分野①',
        '得意分野②',
        '不得意なこと①',
        '不得意なこと②',
        '専門スキル',
        'コミュニケーション力',
        'リーダーシップ',
    ]

    corpus = []
    for _, row in df.iterrows():
        # 存在するカラムだけ結合（NaN は空文字に変換）
        parts = [
            str(row[col]) for col in search_cols
            if col in df.columns and pd.notna(row[col])
        ]
        corpus.append(" ".join(parts))

    return corpus


# ──────────────────────────────────────────────
# 2. TF-IDF ベクトライザの構築（インデックス作成）
# ──────────────────────────────────────────────

def build_tfidf_index(corpus: list[str]):
    """
    コーパス全体に対して TF-IDF 行列を計算し、
    ベクトライザと行列をセットで返す。

    「インデックスを事前に構築しておく」ことで、
    検索のたびに全員分を再計算せず済む（Week4 の DB に相当）。

    Args:
        corpus: build_corpus() が返す文字列リスト

    Returns:
        (vectorizer, tfidf_matrix) のタプル
        - vectorizer : fit 済みの TfidfVectorizer
        - tfidf_matrix: shape=(従業員数, 語彙数) の疎行列
    """
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',   # 文字 n-gram（日本語はスペース分割が効きにくいため）
        ngram_range=(2, 3),   # 2〜3文字のチャンクを特徴量にする
        min_df=1,             # 1件以上に出現した語彙のみ使用
        sublinear_tf=True,    # TF に log スケールを適用（長文ドキュメントの影響を緩和）
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


# ──────────────────────────────────────────────
# 3. TF-IDF スコアリング検索
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

    処理の流れ（Week4 ranking.py と同じ考え方）:
      1. クエリ文字列を同じベクトライザで変換
      2. 全従業員ドキュメントとのコサイン類似度を計算
      3. 部署フィルター適用
      4. スコア降順でソート → 上位 top_n 件を返す

    Args:
        df              : 従業員データの DataFrame
        query           : 検索キーワード（例: "労務管理"）
        vectorizer      : build_tfidf_index() が返した fit 済みベクトライザ
        tfidf_matrix    : build_tfidf_index() が返した TF-IDF 行列
        department      : 部署フィルター（"全部署" で全員対象）
        top_n           : 返す最大件数
        score_threshold : この値未満のスコアは除外（0.0 でしきい値なし）

    Returns:
        スコア付き・スコア降順の DataFrame（'similarity_score' 列が追加される）
    """
    if not query.strip():
        # クエリ空の場合は全件をスコア 0 で返す
        result = df.copy()
        result['similarity_score'] = 0.0
        if department != "全部署":
            result = result[result['所属部署'] == department]
        return result.reset_index(drop=True)

    # ① クエリをベクトル化（fit 済みのベクトライザを使う）
    query_vec = vectorizer.transform([query])

    # ② コサイン類似度を計算（shape: (1, 従業員数)）
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # ③ スコアを DataFrame に付与
    result = df.copy()
    result['similarity_score'] = scores

    # ④ 部署フィルター
    if department != "全部署":
        result = result[result['所属部署'] == department]

    # ⑤ しきい値フィルター
    result = result[result['similarity_score'] >= score_threshold]

    # ⑥ スコア降順ソート → 上位 top_n 件
    result = (
        result
        .sort_values('similarity_score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return result


# ──────────────────────────────────────────────
# 4. ユーティリティ：スコアバーの可視化
# ──────────────────────────────────────────────

def score_to_bar(score: float, width: int = 10) -> str:
    """
    類似度スコア（0〜1）をアスキーアートのバーに変換する。

    例: score=0.75 → "████████░░  0.75"

    Args:
        score : 0.0〜1.0 のスコア
        width : バーの文字数

    Returns:
        バー文字列
    """
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {score:.3f}"


# ──────────────────────────────────────────────
# 5. 動作確認（スタンドアロン実行用）
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ノートブックと同じ方法でデータを読み込む
    EXCEL_PATH = "employee_dummy_data.xlsx"

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

    print("=" * 60)
    print("  TF-IDF スコアリング — 動作確認")
    print("=" * 60)

    # データ読み込み
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="従業員データ", header=1)
    except FileNotFoundError:
        print(f"\n⚠️  {EXCEL_PATH} が見つかりません。")
        print("   このスクリプトを employee_dummy_data.xlsx と同じフォルダに置いてください。")
        exit(1)

    df['専門スキル_数値'] = df['専門スキル'].map(SKILL_LEVEL_MAP)
    df['コミュニケーション力_数値'] = df['コミュニケーション力'].map(SKILL_LEVEL_MAP)
    df['リーダーシップ_数値'] = df['リーダーシップ'].map(SKILL_LEVEL_MAP)

    print(f"\n✅ データ読み込み完了: {len(df)} 名\n")

    # インデックス構築
    corpus = build_corpus(df)
    vectorizer, tfidf_matrix = build_tfidf_index(corpus)
    print(f"✅ TF-IDF インデックス構築完了")
    print(f"   語彙数: {len(vectorizer.vocabulary_):,} 語\n")

    # 検索テスト
    test_queries = ["労務管理", "バックエンド開発", "プロジェクトマネジメント"]

    for query in test_queries:
        print(f"{'─' * 60}")
        print(f"🔍 クエリ: 「{query}」")
        print(f"{'─' * 60}")

        results = search_employees_tfidf(
            df, query, vectorizer, tfidf_matrix,
            top_n=5,
            score_threshold=0.01,
        )

        if results.empty:
            print("  （該当なし）\n")
            continue

        for i, row in results.iterrows():
            bar = score_to_bar(row['similarity_score'])
            print(f"  {i+1}位  {row['氏名']:<10} {row['所属部署']:<12} "
                  f"得意: {row['得意分野①']} / {row['得意分野②']}")
            print(f"       スコア: {bar}")
        print()

    print("=" * 60)
    print("  部署フィルターのテスト（人事部 × 労務管理）")
    print("=" * 60)
    results_dept = search_employees_tfidf(
        df, "労務管理", vectorizer, tfidf_matrix,
        department="人事部",
        top_n=3,
    )
    print(f"\n人事部内の検索結果: {len(results_dept)} 件")
    for i, row in results_dept.iterrows():
        print(f"  {i+1}位  {row['氏名']}  スコア: {row['similarity_score']:.4f}")
