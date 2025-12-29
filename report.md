# 实习五报告
## 刘凌锋部分 2501210091
### 基于以下两种方式实现了从MovieLens数据到图的变换。

- **方式一**：基于用户高分共同观影次数计算用户相似度。

首先我们通过`movieId`的自连接，识别对同一部电影均有高分评论的用户组合（打分$\ge 4.0$），并进行筛选，并按照共同评论的数量作为相似度，筛选高相似用户：
```python
merged_df = pd.merge(high_ratings_df, high_ratings_df, on='movieId', suffixes=('_x', '_y'))
user_pairs_df = merged_df[merged_df['userId_x'] < merged_df['userId_y']]
similarity_df = user_pairs_df.groupby(['userId_x', 'userId_y']).size().reset_index(name='similarity')
similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
```
我们将以上的`similarity_df`最终存成一个csv，它将会含有三列，`userId_x`, `userId_y`, `similarity`，并按照相似度降序排列。这就是我们的一个图。

- **方式二**：基于自己设计的用户画像向量嵌入方式，结合PostgreSQL+PGVector, 实现按照余弦相似度计算的高相似人群识别图。

首先，我们将每一个用户，基于他的过往评论打分，把他映射到一个维度与数据集电影类别数相同的向量。每一个分量对应一个电影的种类，我们计算这个维度的嵌入数值公式为：
$$
s_i = \log (1+\text{number of ratings}) * \text{average rating score}
$$
这代表了一种基于评分数量和平均评分的综合的分数，表征用户有多么喜欢某个类型的电影。

具体实现为：
```python
user_genre_stats = df_exploded.groupby(['userId', 'genres'])['rating'].agg(['mean', 'count']).reset_index()

user_genre_stats['weighted_score'] = np.log1p(user_genre_stats['count']) * user_genre_stats['mean']

user_genre_matrix = user_genre_stats.pivot_table(
    index='userId', 
    columns='genres', 
    values='weighted_score'
).fillna(0)

user_genre_matrix = user_genre_matrix.reindex(columns=all_genres, fill_value=0)
```

然后，我们把用户向量存入启用了pgvector扩展的PostgreSQL表中。
```python
extras.execute_values(cur, "INSERT INTO user_vectors (user_id, profile_vec) VALUES %s", data_to_insert)
```

如果要查询与某用户id最相似的k个用户，可以用如下方式实现：
```python
cur.execute(
    "SELECT user_id, 1 - (profile_vec <=> %s) AS cosine_similarity FROM user_vectors WHERE user_id != %s ORDER BY profile_vec <=> %s LIMIT %s;",
    (target_vector, target_user_id, target_vector, top_k)
)
results = cur.fetchall()
```

如果要基于本数据库生成一个高相似用户图，可以按照如下的方法实现：
```python
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
similarity_df = pd.read_sql_query(query, conn, params=(similarity_threshold,))
```

同样，把这个得到的`similarity_df`转为csv存储，就得到一个与方式1中格式一致的表。这里的区别在于，`similarity`列将会是一个取值在$[\text{similarity\_threshold},\;1]$以内的数。