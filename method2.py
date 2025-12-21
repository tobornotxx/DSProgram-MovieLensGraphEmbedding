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

def generate_similarity_edge_list(user_vectors, similarity_threshold=0.75):
    """
    计算所有用户向量之间的余弦相似度，并生成一个带权的边列表。

    Args:
        user_vectors (dict): {userId: numpy_vector} 的字典。
        similarity_threshold (float): 只保留相似度高于此阈值的边。

    Returns:
        pandas.DataFrame: 包含 'user_id_x', 'user_id_y', 'similarity' 的DataFrame。
    """
    print(f"\n--- Generate graph edge list with method 2. ---")
    
    # 将字典转换为ID列表和向量矩阵，确保顺序一致
    user_ids = list(user_vectors.keys())
    vectors_matrix = np.array(list(user_vectors.values()))
    
    # 高效计算所有对之间的余弦相似度
    print("Computing Cosine similarity matrix")
    cosine_sim_matrix = cosine_similarity(vectors_matrix)
    
    # 构建边列表
    edge_list = []
    num_users = len(user_ids)
    
    # 遍历相似度矩阵的上三角部分，避免重复和自我连接
    for i in range(num_users):
        for j in range(i + 1, num_users):
            similarity = cosine_sim_matrix[i, j]
            
            # 应用阈值，过滤掉权重较低的边
            if similarity >= similarity_threshold:
                edge_list.append({
                    'userId_x': user_ids[i],
                    'userId_y': user_ids[j],
                    'similarity': similarity
                })
    
    if not edge_list:
        print(f"Warning: with {similarity_threshold} as threshold, no edges were generated.")
        return pd.DataFrame(columns=['userId_x', 'userId_y', 'similarity'])

    # 转换为DataFrame并排序
    similarity_df = pd.DataFrame(edge_list)
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
    
    print(f"Under condition of Threshold > {similarity_threshold}, generated {len(similarity_df)} edges.")
    print("Top 10 pairs with highest similarity:")
    print(similarity_df.head(10))
    
    return similarity_df

if __name__ == "__main__":
    RATINGS_FILE = './ml-latest-small/ratings.csv'
    MOVIES_FILE = './ml-latest-small/movies.csv'

    vectors, genres = prepare_user_vectors_weighted_and_normalized(RATINGS_FILE, MOVIES_FILE)
    
    if vectors:
        # 1. 执行数据库查询
        store_and_query_vectors(vectors, target_user_id=1, top_k=5)

        # 2. 生成图的边列表文件
        # 作业要求“过滤掉比较低的边”，这里的阈值就很重要。
        user_similarity_graph_m2 = generate_similarity_edge_list(vectors, similarity_threshold=0.50)
        
        if not user_similarity_graph_m2.empty:
            output_path_m2 = 'user_similarity_method2.csv'
            user_similarity_graph_m2.to_csv(output_path_m2, index=False)
            print(f"\nMethod 2 Graph data saved to file: {output_path_m2}")