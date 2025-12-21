import pandas as pd

def calculate_co_view_similarity(ratings_path, movies_path, rating_threshold=4.0):
    """
    基于用户高分共同观影次数计算用户相似度 (方式1)。

    Args:
        ratings_path (str): ratings.csv 文件路径。
        movies_path (str): movies.csv 文件路径。
        rating_threshold (float): 定义高分的阈值。

    Returns:
        pandas.DataFrame: 包含用户对和相似度（共同观影次数）的DataFrame。
    """
    print("--- Method1: Similarity computation based on shared movie watches. ---")

    # 1. 加载数据
    try:
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        print("Data Loaded")
        print("Ratings Data Head:")
        print(ratings_df.head())
        print("\nMovies Data Head:")
        print(movies_df.head())
    except FileNotFoundError as e:
        print(f"Data File Not Found {e}")
        return None

    # 2. 筛选高分评价
    high_ratings_df = ratings_df[ratings_df['rating'] >= rating_threshold]
    print(f"\nFiltered reviews, preserving reviews with score >= {rating_threshold}, {len(high_ratings_df)} in total.")

    # 3. 通过电影ID自连接，找到观看同一电影的用户对
    # 这是实现二部图投影的核心步骤
    merged_df = pd.merge(high_ratings_df, high_ratings_df, on='movieId', suffixes=('_x', '_y'))
    
    # 4. 过滤掉自己与自己的匹配，以及重复的用户对
    # (user1, user2) 和 (user2, user1) 只保留一种
    user_pairs_df = merged_df[merged_df['userId_x'] < merged_df['userId_y']]

    # 5. 分组并计数，计算每对用户的共同观影次数
    similarity_df = user_pairs_df.groupby(['userId_x', 'userId_y']).size().reset_index(name='similarity')
    
    # 6. 按相似度降序排序
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

    print("\nTop 10 pairs with highest similarity (shared movie watching):")
    print(similarity_df.head(10))
    
    return similarity_df

if __name__ == "__main__":
    # 数据文件路径 (请根据你的实际情况修改)
    RATINGS_FILE = './ml-latest-small/ratings.csv'
    MOVIES_FILE = './ml-latest-small/movies.csv'
    
    # 执行计算
    user_similarity_graph = calculate_co_view_similarity(RATINGS_FILE, MOVIES_FILE)
    
    # 7. 保存结果为CSV文件
    if user_similarity_graph is not None:
        output_path = 'user_similarity_method1.csv'
        user_similarity_graph.to_csv(output_path, index=False)
        print(f"\nResult saved to file: {output_path}")