# method2_pgvector.py
from locale import normalize
import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def get_db_config():
    """从环境变量中安全地读取数据库配置。"""
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("GRAPH_DB_NAME"),
        "user": os.getenv("GRAPH_DB_USER"),
        "password": os.getenv("GRAPH_DB_PASS")
    }
    # 检查必要的配置是否存在
    if not all([config["dbname"], config["user"], config["password"]]):
        raise ValueError("Environment variable (GRAPH_DB_NAME, GRAPH_DB_USER, GRAPH_DB_PASS) not found.")
    return config

def prepare_user_vectors_weighted_and_normalized(ratings_path, movies_path):
    """
    根据用户对各类型的加权评分（log(1+count)*avg_rating）生成用户画像向量

    Args:
        ratings_path (str): ratings.csv 文件路径。
        movies_path (str): movies.csv 文件路径。

    Returns:
        - dict: {userId: numpy_vector}
        - list: 所有类型的有序列表，作为向量维度的参考。
    """
    print("--- Start Method 2, weighted average score as user portrait vector. ---")

    # 1. 加载和预处理数据
    try:
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
    except FileNotFoundError as e:
        print(f"Error: Data file not found- {e}")
        return None, None
    
    movies_df['genres'] = movies_df['genres'].str.split('|')
    all_genres = sorted(list(set(g for G in movies_df['genres'] for g in G)))
    num_genres = len(all_genres)
    genre_map = {genre: i for i, genre in enumerate(all_genres)}
    print(f"Detected {num_genres} distinct movie genres")
    
    # 2. 合并并展开数据
    df = pd.merge(ratings_df, movies_df, on='movieId')
    df_exploded = df.explode('genres')

    # 3. 计算每个用户对每个类型的平均分和评分次数
    # 使用 agg() 函数一次性计算 mean 和 count
    user_genre_stats = df_exploded.groupby(['userId', 'genres'])['rating'].agg(['mean', 'count']).reset_index()

    # 4. 应用加权公式: log(1 + count) * mean
    user_genre_stats['weighted_score'] = np.log1p(user_genre_stats['count']) * user_genre_stats['mean']
    
    # 5. 构建用户向量矩阵
    user_genre_matrix = user_genre_stats.pivot_table(
        index='userId', 
        columns='genres', 
        values='weighted_score'
    ).fillna(0)
    user_genre_matrix = user_genre_matrix.reindex(columns=all_genres, fill_value=0)

    # 6. 存入字典
    user_vectors = {}
    for user_id, row in user_genre_matrix.iterrows():

        # vec = row.to_numpy(dtype=np.float32)

        normalized_vec = row.to_numpy(dtype=np.float32)
        
        # # L2 标准化
        # norm = np.linalg.norm(vec)
        # if norm > 0:
        #     normalized_vec = vec / norm
        # else:
        #     normalized_vec = vec # 零向量保持不变
        
        user_vectors[user_id] = normalized_vec

    print(f"Generated portrait vector for {len(user_vectors)} users")

    # (可选) 展示一个用户的画像和向量
    sample_user_id = 1
    if sample_user_id in user_genre_matrix.index:
        user_profile = user_genre_matrix.loc[sample_user_id]
        top_k_genres = user_profile[user_profile > 0].sort_values(ascending=False).head(5)
        print(f"\nExample User {sample_user_id} top-5 genres based on weighted average score:")
        print(top_k_genres)
        print(f"\nExample User {sample_user_id} vector :")
        print(user_vectors[sample_user_id])

    return user_vectors, all_genres


def store_and_query_vectors(user_vectors, target_user_id=1, top_k=5):
    """将向量存入 PostgreSQL 并执行相似度查询。"""
    if not user_vectors:
        print("No available user vector. Program quit.")
        return
    conn = None
    try:
        db_config = get_db_config()
        print("\nConnecting to PostgreSQL DB.")
        conn = psycopg2.connect(**db_config)
        register_vector(conn)
        cur = conn.cursor()
        print("DB Connected.")
        print("Cleaning user_vector table...")
        cur.execute("TRUNCATE TABLE user_vectors;")
        print(f"Inserting {len(user_vectors)} user vectors into database.")
        data_to_insert = [(user_id, vec) for user_id, vec in user_vectors.items()]
        extras.execute_values(cur, "INSERT INTO user_vectors (user_id, profile_vec) VALUES %s", data_to_insert)
        conn.commit()
        print("Done Insert.")

        target_vector = user_vectors.get(target_user_id)
        if target_vector is None:
            print(f"Error: User {target_user_id} does not exist.")
            return

        print(f"\n Searching {top_k} most similar users to user {target_user_id}.")
        cur.execute(
            "SELECT user_id, 1 - (profile_vec <=> %s) AS cosine_similarity FROM user_vectors WHERE user_id != %s ORDER BY profile_vec <=> %s LIMIT %s;",
            (target_vector, target_user_id, target_vector, top_k)
        )
        results = cur.fetchall()
        
        print("\n--- Search Result ---")
        print(f"{'User ID':<10} | {'Cosine Similarity':<20}")
        print("-" * 35)
        for row in results:
            print(f"{row[0]:<10} | {row[1]:<20.4f}")
        print("---------------------\n")

    except (psycopg2.OperationalError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unknown Error: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close(); print("Database connection closed.")


def generate_similarity_edge_list_with_sql(similarity_threshold=0.50):
    """
    直接使用 SQL 查询在数据库中计算用户相似度，并获取高于阈值的边列表。

    Args:
        similarity_threshold (float): 只保留相似度高于此阈值的边。

    Returns:
        pandas.DataFrame: 包含 'user_id_x', 'user_id_y', 'similarity' 的DataFrame，
                          如果出错或没有数据则返回 None。
    """
    print(f"\n--- Generate graph edge list using SQL (Recommended Method). ---")
    
    conn = None
    query = """
        SELECT
            t1.user_id AS userId_x,
            t2.user_id AS userId_y,
            1 - (t1.profile_vec <=> t2.profile_vec) AS similarity
        FROM
            user_vectors t1
        JOIN
            user_vectors t2 ON t1.user_id < t2.user_id
        WHERE
            1 - (t1.profile_vec <=> t2.profile_vec) >= %s
        ORDER BY
            similarity DESC;
    """
    
    try:
        db_config = get_db_config()
        print("Connecting to PostgreSQL DB to compute similarities.")
        conn = psycopg2.connect(**db_config)
        register_vector(conn) # 确保 pgvector 类型被识别
        
        print(f"Executing query with similarity threshold >= {similarity_threshold}...")
        
        # 使用 pandas 直接从 SQL 查询读取数据到 DataFrame
        similarity_df = pd.read_sql_query(query, conn, params=(similarity_threshold,))
        
        if similarity_df.empty:
            print(f"Warning: with {similarity_threshold} as threshold, no edges were generated from DB.")
        else:
            print(f"Successfully generated {len(similarity_df)} edges from database.")
            print("Top 10 pairs with highest similarity:")
            print(similarity_df.head(10))
            
        return similarity_df

    except (psycopg2.OperationalError, ValueError) as e:
        print(f"Error connecting to or querying the database: {e}")
        return None
    finally:
        if conn: conn.close(); print("Database connection closed.")

if __name__ == "__main__":
    RATINGS_FILE = './ml-latest-small/ratings.csv'
    MOVIES_FILE = './ml-latest-small/movies.csv'

    vectors, genres = prepare_user_vectors_weighted_and_normalized(RATINGS_FILE, MOVIES_FILE)
    
    if vectors:
        # 1. 执行数据库查询
        store_and_query_vectors(vectors, target_user_id=1, top_k=5)

    print("\nStarting process to generate edge list directly from database using SQL...")
    user_similarity_graph_m2 = generate_similarity_edge_list_with_sql(similarity_threshold=0.50)
    
    if user_similarity_graph_m2 is not None and not user_similarity_graph_m2.empty:
        output_path_m2 = 'user_similarity_method2.csv'
        user_similarity_graph_m2.to_csv(output_path_m2, index=False)
        print(f"\nMethod 2 Graph data saved to file: {output_path_m2}")