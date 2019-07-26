#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18 9:53 AM
# @Author  : R
# @File    : TMDB_predict_2.py
# @Software: PyCharm

# @Software: PyCharm


# coding: utf-8

# # Kaggle for TMDB

# In[1]:


import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from collections import Counter
warnings.filterwarnings('ignore')


# get_ipython().run_line_magic('matplotlib', 'inline')

# Data description
# id：每部电影的唯一标志
# belongs_to_collection:json格式下每部电影的tmdb id， 电影名、电影海报和电影背景的URL
# budget:电影预算，数值为0表示未知
# genres：电影风格列表，json文件，包含id、name
# homepage：电影官方主页的URL
# imdb_id:该电影在imdb数据库中的唯一id标志
# original_language：电影制作的原始语言，长度为2的字符串
# original_title：电影的原始名称，可能与belong_to_collection中的名称不同
# overview： 剧情摘要
# popularity： 电影的受欢迎程度，float数值表示
# poster_path: 电影海报的URL
# production_companies：json格式，电影制造公司的id、name
# production_countries：json格式，电影制造国家 2字符简称、全称
# release_date：电影上映时间
# runtime：电影时长
# spoken_languages：电影语言版本，json格式
# status:电影是否已经发布
# tagline： 电影的标语
# title: 电影的英文名称
# keywords：电影关键字，json格式
# cast: json格式，演员列表，包括id，name，性别等
# crew：电影制作人员的信息，包括导演，作者等
# revenue：总收入，待预测值
# # EDA

# EDA已做

# 特征工程以及预测
# 利用两个额外的数据集合
# 1.TMDB Competition Additional Features:本数据包含新的三个特征popularity2、rating、totalVotes
# 2.TMDB Competition Additional Training Data：额外的2000个训练数据，没有给定训练集中所有的属性

# In[52]:


# Feature Engineering & Prediction

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 数据预处理函数，包括将非数值型属性转化为数值型
def prepare(df):
    global json_cols
    global train_dict

    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), "release_year"] += 1900

    # 获取发行日期的星期、季度信息
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    # 对rating、totalVotes属性进行填充
    rating_na = df.groupby(["release_year", "original_language"])['rating'].mean().reset_index()
    df[df.rating.isna()]['rating'] = df.merge(rating_na, how='left', on=["release_year", "original_language"])
    vote_count_na = df.groupby(["release_year", "original_language"])['totalVotes'].mean().reset_index()
    df[df.totalVotes.isna()]['totalVotes'] = df.merge(vote_count_na, how='left',
                                                      on=["release_year", "original_language"])
    # df['rating'] = df['rating'].fillna(1.5)
    # df['totalVotes'] = df['totalVotes'].fillna(6)

    # 构建一个新属性，weightRating
    df['weightedRating'] = (df['rating'] * df['totalVotes'] + 6.367 * 1000) / (df['totalVotes'] + 1000)

    # 考虑到不同时期的面额意义不同，对其进行“通货膨胀”，通货膨胀比例为1.8%/年
    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (
                2018 - df['release_year'])  # Inflation simple formula
    df['budget'] = np.log1p(df['budget'])

    # 对crew、cast属性中人员性别构成进行统计
    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    # 对belongs_to_collection、Keywords、cast进行统计
    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df['_collection_name'].fillna('')))
    df['_collection_name'] = le.transform(df['_collection_name'].fillna('').astype(str))
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)


    df['_popularity_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')
    df['_budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['_budget_popularity_ratio'] = df['budget'] / df['popularity']
    # df['_budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])
    # df['_releaseYear_popularity_ratio'] = df['release_year'] / df['popularity']
    # df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']

    df['_popularity_totalVotes_ratio'] = df['totalVotes'] / df['popularity']
    df['_rating_popularity_ratio'] = df['rating'] / df['popularity']
    df['_rating_totalVotes_ratio'] = df['totalVotes'] / df['rating']
    # df['_totalVotes_releaseYear_ratio'] = df['totalVotes'] / df['release_year']
    df['_budget_rating_ratio'] = df['budget'] / df['rating']
    df['_runtime_rating_ratio'] = df['runtime'] / df['rating']
    df['_budget_totalVotes_ratio'] = df['budget'] / df['totalVotes']

    # 对是否有homepage分类
    df['has_homepage'] = 1
    df.loc[pd.isnull(df['homepage']), "has_homepage"] = 0

    # 对belongs_to_collection是否为空分类
    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), "isbelongs_to_collectionNA"] = 1

    # 对tagline是否为空分类
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0, "isTaglineNA"] = 1

    # 对original——langues是否为English判定
    df['isOriginalLanguageEng'] = 0
    df.loc[df['original_language'] == "en", "isOriginalLanguageEng"] = 1

    # 对电影名是否不同判定
    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], "isTitleDifferent"] = 0

    # 对电影是否上映判定
    df['isMovieReleased'] = 1
    df.loc[df['status'] != "Released", "isMovieReleased"] = 0

    # 电影是否有摘要
    df['isOverviewNA'] = 0
    df.loc[pd.isnull(df['overview']), 'isOverviewNA'] = 1

    # 获取collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x: np.nan if len(x) == 0 else x[0]['id'])

    # 对original——title等属性统计长度
    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()

    # 对title、overview、tagline统计长度或个数
    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    df['len_title'] = df['title'].fillna('').apply(lambda x: len(str(x)))

    # 对genres、production_conpany、country、cast、crew、spoken_languages统计
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['cast_count'] = df['cast'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x))
    df['spoken_languages_count'] = df['spoken_languages'].apply(lambda x: len(x))
    df['genres_count'] = df['genres'].apply(lambda x: len(x))

    # 进行按年分组计算均值填充
    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')
    df['meantotalVotesByYear'] = df.groupby("release_year")["totalVotes"].aggregate('mean')
    df['meanTotalVotesByRating'] = df.groupby("rating")["totalVotes"].aggregate('mean')
    df['medianBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('median')

    df['_popularity_theatrical_ratio'] = df['theatrical'] / df['popularity']
    df['_budget_theatrical_ratio'] = df['budget'] / df['theatrical']

    # runtime
    df['runtime_cat_min_60'] = df['runtime'].apply(lambda x: 1 if (x <= 60) else 0)
    df['runtime_cat_61_80'] = df['runtime'].apply(lambda x: 1 if (x > 60) & (x <= 80) else 0)
    df['runtime_cat_81_100'] = df['runtime'].apply(lambda x: 1 if (x > 80) & (x <= 100) else 0)
    df['runtime_cat_101_120'] = df['runtime'].apply(lambda x: 1 if (x > 100) & (x <= 120) else 0)
    df['runtime_cat_121_140'] = df['runtime'].apply(lambda x: 1 if (x > 120) & (x <= 140) else 0)
    df['runtime_cat_141_170'] = df['runtime'].apply(lambda x: 1 if (x > 140) & (x <= 170) else 0)
    df['runtime_cat_171_max'] = df['runtime'].apply(lambda x: 1 if (x >= 170) else 0)

    lang = df['original_language']
    df_more_17_samples = [x[0] for x in Counter(pd.DataFrame(lang).stack()).most_common(17)]
    for col in df_more_17_samples:
        df[col] = df['original_language'].apply(lambda x: 1 if x == col else 0)

    for col in range(1, 12):
        df['month' + str(col)] = df['release_month'].apply(lambda x: 1 if x == col else 0)
    # feature engeneering : Release date per quarter one hot encoding
    for col in range(1, 4):
        df['quarter' + str(col)] = df['release_quarter'].apply(lambda x: 1 if x == col else 0)
    for col in range(1, 7):
        df['dayofweek' + str(col)] = df['release_dayofweek'].apply(lambda x: 1 if x == col else 0)

# 新加入属性
    df['is_release_day_of_1'] = 0
    df.loc[df['release_day'] == 1, 'is_release_day_of_1'] = 1

    df['is_release_day_of_15'] = 0
    df.loc[df['release_day'] == 15, 'is_release_day_of_15'] = 1

# 新属性加入
#     df['popularity2'] = np.log1p(df['popularity2'])
#     df['popularity'] = np.log1p(df['popularity'])
#     for col in range(1, 32):
#         df['release_day' + str(col)] = df['release_day'].apply(lambda x: 1 if x == col else 0)

    df['is_release_day_of_31'] = 0
    df.loc[df['release_day'] == 31, 'is_release_day_of_15'] = 1
    # popularity
    # df['popularity_cat_25'] = df['popularity'].apply(lambda x: 1 if (x <= 25) else 0)
    # df['popularity_cat_26_50'] = df['popularity'].apply(lambda x: 1 if (x > 25) & (x <= 50) else 0)
    # df['popularity_cat_51_100'] = df['popularity'].apply(lambda x: 1 if (x > 50) & (x <= 100) else 0)
    # df['popularity_cat_101_150'] = df['popularity'].apply(lambda x: 1 if (x > 100) & (x <= 150) else 0)
    # df['popularity_cat_151_200'] = df['popularity'].apply(lambda x: 1 if (x > 150) & (x <= 200) else 0)
    # df['popularity_cat_201_max'] = df['popularity'].apply(lambda x: 1 if (x >= 200) else 0)
    #
    # df['_runtime_totalVotes_ratio'] = df['runtime'] / df['totalVotes']
    # df['_runtime_popularity_ratio'] = df['runtime'] / df['popularity']
    #
    # df['_rating_theatrical_ratio'] = df['theatrical'] / df['rating']
    # df['_totalVotes_theatrical_ratio'] = df['theatrical'] / df['totalVotes']
    # df['_budget_mean_year'] = df['budget'] / df.groupby("release_year")["budget"].transform('mean')
    # df['_runtime_mean_year'] = df['runtime'] / df.groupby("release_year")["runtime"].transform('mean')
    # df['_rating_mean_year'] = df['rating'] / df.groupby("release_year")["rating"].transform('mean')
    # df['_totalVotes_mean_year'] = df['totalVotes'] / df.groupby("release_year")["totalVotes"].transform('mean')





    # 对某些json属性，具有多个值的，进行类似‘one-hot编码’
    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies', 'Keywords']:
        df[col] = df[col].map(lambda x: sorted(
            list(set([n if n in train_dict[col] else col + '_etc' for n in [d['name'] for d in x]])))).map(
            lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)

    # 删除非数值属性和暂时未提出有用信息的属性
    df.drop(['genres_etc'], axis=1, inplace=True)
    df = df.drop(['belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview','runtime'
                     , 'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages'
                     , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title', 'tagline',
                  'collection_id'
                  ], axis=1)

    # 填充缺失值
    df.fillna(value=0.0, inplace=True)

    return df


# 对train中的某些数据手动处理
# 处理包括budget、revenue
# 对budget远小于revenue的情况统计，对其进行处理
# 处理原则，对于可以查询到的信息，进行真实数据填充，否则取当年同期同类型电影的均值
train = pd.read_csv('train.csv')
train.loc[train['id'] == 16, 'revenue'] = 192864  # Skinning
train.loc[train['id'] == 90, 'budget'] = 30000000  # Sommersby
train.loc[train['id'] == 118, 'budget'] = 60000000  # Wild Hogs
train.loc[train['id'] == 149, 'budget'] = 18000000  # Beethoven
train.loc[train['id'] == 313, 'revenue'] = 12000000  # The Cookout
train.loc[train['id'] == 451, 'revenue'] = 12000000  # Chasing Liberty
train.loc[train['id'] == 464, 'budget'] = 20000000  # Parenthood
train.loc[train['id'] == 470, 'budget'] = 13000000  # The Karate Kid, Part II
train.loc[train['id'] == 513, 'budget'] = 930000  # From Prada to Nada
train.loc[train['id'] == 797, 'budget'] = 8000000  # Welcome to Dongmakgol
train.loc[train['id'] == 819, 'budget'] = 90000000  # Alvin and the Chipmunks: The Road Chip
train.loc[train['id'] == 850, 'budget'] = 90000000  # Modern Times
train.loc[train['id'] == 1007, 'budget'] = 2  # Zyzzyx Road
train.loc[train['id'] == 1112, 'budget'] = 7500000  # An Officer and a Gentleman
train.loc[train['id'] == 1131, 'budget'] = 4300000  # Smokey and the Bandit
train.loc[train['id'] == 1359, 'budget'] = 10000000  # Stir Crazy
train.loc[train['id'] == 1542, 'budget'] = 1  # All at Once
train.loc[train['id'] == 1570, 'budget'] = 15800000  # Crocodile Dundee II
train.loc[train['id'] == 1571, 'budget'] = 4000000  # Lady and the Tramp
train.loc[train['id'] == 1714, 'budget'] = 46000000  # The Recruit
train.loc[train['id'] == 1721, 'budget'] = 17500000  # Cocoon
train.loc[train['id'] == 1865, 'revenue'] = 25000000  # Scooby-Doo 2: Monsters Unleashed
train.loc[train['id'] == 1885, 'budget'] = 12  # In the Cut
train.loc[train['id'] == 2091, 'budget'] = 10  # Deadfall
train.loc[train['id'] == 2268, 'budget'] = 17500000  # Madea Goes to Jail budget
train.loc[train['id'] == 2491, 'budget'] = 6  # Never Talk to Strangers
train.loc[train['id'] == 2602, 'budget'] = 31000000  # Mr. Holland's Opus
train.loc[train['id'] == 2612, 'budget'] = 15000000  # Field of Dreams
train.loc[train['id'] == 2696, 'budget'] = 10000000  # Nurse 3-D
train.loc[train['id'] == 2801, 'budget'] = 10000000  # Fracture
train.loc[train['id'] == 335, 'budget'] = 2
train.loc[train['id'] == 348, 'budget'] = 12
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 1100000
train.loc[train['id'] == 640, 'budget'] = 6
train.loc[train['id'] == 696, 'budget'] = 1
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 850, 'budget'] = 1500000
train.loc[train['id'] == 1199, 'budget'] = 5
train.loc[train['id'] == 1282, 'budget'] = 9  # Death at a Funeral
train.loc[train['id'] == 1347, 'budget'] = 1
train.loc[train['id'] == 1755, 'budget'] = 2
train.loc[train['id'] == 1801, 'budget'] = 5
train.loc[train['id'] == 1918, 'budget'] = 592
train.loc[train['id'] == 2033, 'budget'] = 4
train.loc[train['id'] == 2118, 'budget'] = 344
train.loc[train['id'] == 2252, 'budget'] = 130
train.loc[train['id'] == 2256, 'budget'] = 1
train.loc[train['id'] == 2696, 'budget'] = 10000000

# test异常处理
test = pd.read_csv('test.csv')
# Clean Data
test.loc[test['id'] == 6733, 'budget'] = 5000000
test.loc[test['id'] == 3889, 'budget'] = 15000000
test.loc[test['id'] == 6683, 'budget'] = 50000000
test.loc[test['id'] == 5704, 'budget'] = 4300000
test.loc[test['id'] == 6109, 'budget'] = 281756
test.loc[test['id'] == 7242, 'budget'] = 10000000
test.loc[test['id'] == 7021, 'budget'] = 17540562  # Two Is a Family
test.loc[test['id'] == 5591, 'budget'] = 4000000  # The Orphanage
test.loc[test['id'] == 4282, 'budget'] = 20000000  # Big Top Pee-wee
test.loc[test['id'] == 3033, 'budget'] = 250
test.loc[test['id'] == 3051, 'budget'] = 50
test.loc[test['id'] == 3084, 'budget'] = 337
test.loc[test['id'] == 3224, 'budget'] = 4
test.loc[test['id'] == 3594, 'budget'] = 25
test.loc[test['id'] == 3619, 'budget'] = 500
test.loc[test['id'] == 3831, 'budget'] = 3
test.loc[test['id'] == 3935, 'budget'] = 500
test.loc[test['id'] == 4049, 'budget'] = 995946
test.loc[test['id'] == 4424, 'budget'] = 3
test.loc[test['id'] == 4460, 'budget'] = 8
test.loc[test['id'] == 4555, 'budget'] = 1200000
test.loc[test['id'] == 4624, 'budget'] = 30
test.loc[test['id'] == 4645, 'budget'] = 500
test.loc[test['id'] == 4709, 'budget'] = 450
test.loc[test['id'] == 4839, 'budget'] = 7
test.loc[test['id'] == 3125, 'budget'] = 25
test.loc[test['id'] == 3142, 'budget'] = 1
test.loc[test['id'] == 3201, 'budget'] = 450
test.loc[test['id'] == 3222, 'budget'] = 6
test.loc[test['id'] == 3545, 'budget'] = 38
test.loc[test['id'] == 3670, 'budget'] = 18
test.loc[test['id'] == 3792, 'budget'] = 19
test.loc[test['id'] == 3881, 'budget'] = 7
test.loc[test['id'] == 3969, 'budget'] = 400
test.loc[test['id'] == 4196, 'budget'] = 6
test.loc[test['id'] == 4221, 'budget'] = 11
test.loc[test['id'] == 4222, 'budget'] = 500
test.loc[test['id'] == 4285, 'budget'] = 11
test.loc[test['id'] == 4319, 'budget'] = 1
test.loc[test['id'] == 4639, 'budget'] = 10
test.loc[test['id'] == 4719, 'budget'] = 45
test.loc[test['id'] == 4822, 'budget'] = 22
test.loc[test['id'] == 4829, 'budget'] = 20
test.loc[test['id'] == 4969, 'budget'] = 20
test.loc[test['id'] == 5021, 'budget'] = 40
test.loc[test['id'] == 5035, 'budget'] = 1
test.loc[test['id'] == 5063, 'budget'] = 14
test.loc[test['id'] == 5119, 'budget'] = 2
test.loc[test['id'] == 5214, 'budget'] = 30
test.loc[test['id'] == 5221, 'budget'] = 50
test.loc[test['id'] == 4903, 'budget'] = 15
test.loc[test['id'] == 4983, 'budget'] = 3
test.loc[test['id'] == 5102, 'budget'] = 28
test.loc[test['id'] == 5217, 'budget'] = 75
test.loc[test['id'] == 5224, 'budget'] = 3
test.loc[test['id'] == 5469, 'budget'] = 20
test.loc[test['id'] == 5840, 'budget'] = 1
test.loc[test['id'] == 5960, 'budget'] = 30
test.loc[test['id'] == 6506, 'budget'] = 11
test.loc[test['id'] == 6553, 'budget'] = 280
test.loc[test['id'] == 6561, 'budget'] = 7
test.loc[test['id'] == 6582, 'budget'] = 218
test.loc[test['id'] == 6638, 'budget'] = 5
test.loc[test['id'] == 6749, 'budget'] = 8
test.loc[test['id'] == 6759, 'budget'] = 50
test.loc[test['id'] == 6856, 'budget'] = 10
test.loc[test['id'] == 6858, 'budget'] = 100
test.loc[test['id'] == 6876, 'budget'] = 250
test.loc[test['id'] == 6972, 'budget'] = 1
test.loc[test['id'] == 7079, 'budget'] = 8000000
test.loc[test['id'] == 7150, 'budget'] = 118
test.loc[test['id'] == 6506, 'budget'] = 118
test.loc[test['id'] == 7225, 'budget'] = 6
test.loc[test['id'] == 7231, 'budget'] = 85
test.loc[test['id'] == 5222, 'budget'] = 5
test.loc[test['id'] == 5322, 'budget'] = 90
test.loc[test['id'] == 5350, 'budget'] = 70
test.loc[test['id'] == 5378, 'budget'] = 10
test.loc[test['id'] == 5545, 'budget'] = 80
test.loc[test['id'] == 5810, 'budget'] = 8
test.loc[test['id'] == 5926, 'budget'] = 300
test.loc[test['id'] == 5927, 'budget'] = 4
test.loc[test['id'] == 5986, 'budget'] = 1
test.loc[test['id'] == 6053, 'budget'] = 20
test.loc[test['id'] == 6104, 'budget'] = 1
test.loc[test['id'] == 6130, 'budget'] = 30
test.loc[test['id'] == 6301, 'budget'] = 150
test.loc[test['id'] == 6276, 'budget'] = 100
test.loc[test['id'] == 6473, 'budget'] = 100
test.loc[test['id'] == 6842, 'budget'] = 30


release_dates = pd.read_csv('release_dates_per_country.csv')
release_dates['id'] = range(1,7399)
release_dates.drop(['original_title','title'],axis = 1,inplace = True)
release_dates.index = release_dates['id']
train = pd.merge(train, release_dates, how='left', on=['id'])
test = pd.merge(test, release_dates, how='left', on=['id'])

test['revenue'] = np.nan
# 将从TMDB下载的其他特征进行合并
train = pd.merge(train, pd.read_csv('TrainAdditionalFeatures.csv'),
                 how='left', on=['imdb_id'])
test = pd.merge(test, pd.read_csv('TestAdditionalFeatures.csv'),
                how='left', on=['imdb_id'])

# 添加额外的训练集，2000条
additionalTrainData = pd.read_csv('additionalTrainData.csv')
additionalTrainData['release_date'] = additionalTrainData['release_date'].astype('str')
additionalTrainData['release_date'] = additionalTrainData['release_date'].str.replace('-', '/')
train = pd.concat([train, additionalTrainData])
print('train.columns:', train.columns)
print('train.shape:', train.shape)

# 根据EDA分析结果，对revenue做数据平滑处理
# train['revenue'] = np.log1p(train['revenue'])
y = train['revenue'].values

# json 格式属性列
json_cols = ['genres', 'production_companies', 'production_countries',
             'spoken_languages', 'Keywords', 'cast', 'crew']


# 将json格式属性转化为dict格式
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


for col in tqdm(json_cols + ['belongs_to_collection']):
    train[col] = train[col].apply(lambda x: get_dictionary(x))
    test[col] = test[col].apply(lambda x: get_dictionary(x))


# 统计json格式属性中各个类别出现的次数
def get_json_dict(df):
    global json_cols
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = df[e_col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


train_dict = get_json_dict(train)
test_dict = get_json_dict(test)

# 对json格式列，移除异常类别和出现次数过低类别
# 首先删除非train和test共同出现的类别
# 再删除数量少于10的类别
for col in json_cols:
    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))

    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove):
        if train_dict[col][i] < 10 or i == '':
            remove += [i]

    for i in remove:
        if i in train_dict[col]:
            del train_dict[col][i]
        if i in test_dict[col]:
            del test_dict[col][i]

# 对数据进行预处理
all_data = prepare(pd.concat([train, test]).reset_index(drop=True))
train = all_data.loc[:train.shape[0] - 1, :]
test = all_data.loc[train.shape[0]:, :]
print(train.columns)
train.head()

# In[53]:


random_seed = 2019

features = list(train.columns)
features = [i for i in features if i != 'id' and i != 'revenue']

# 模型构建和预测
from sklearn.metrics import mean_squared_error
def score(data, y):
    validation_res = pd.DataFrame(
    {"id": data["id"].values,
     "transactionrevenue": data["revenue"].values,
     "predictedrevenue": np.expm1(y)})

    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values),
                                     np.log1p(validation_res["predictedrevenue"].values)))


from sklearn.model_selection import GroupKFold


class KFoldValidation():
    def __init__(self, data, n_splits=10):
        unique_vis = np.array(sorted(data['id'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])

        self.fold_ids = []
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
                ids[data['id'].astype(str).isin(unique_vis[val_vis])]
            ])

    def validate(self, train, test, features, model, name="", prepare_stacking=False,
                 fit_params={"early_stopping_rounds": 500, "verbose": 100, "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0

        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN

        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["revenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["revenue"].iloc[val])

            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)

            if len(model.feature_importances_) == len(features):
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions) ** 0.5)

            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            if prepare_stacking:
                train[name].iloc[val] = predictions

                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)

        print("Final score: ", full_score)
        return full_score

Kfolder = KFoldValidation(train)

lgbmodel = lgb.LGBMRegressor(n_estimators=10000,
                             objective='regression',
                             metric='rmse',
                             max_depth = 5,
                             num_leaves=30,
                             min_child_samples=100,
                             learning_rate=0.01,
                             boosting = 'gbdt',
                             min_data_in_leaf= 10,
                             feature_fraction = 0.9,
                             bagging_freq = 1,
                             bagging_fraction = 0.9,
                             importance_type='gain',
                             lambda_l1 = 0.2,
                             bagging_seed=random_seed,
                             subsample=.8,
                             colsample_bytree=.9,
                             use_best_model=True)

Kfolder.validate(train, test, features , lgbmodel, name="lgbfinal", prepare_stacking=True)

lgbmodel.FI.mean(axis=1).sort_values()[180:250].plot(kind="barh",title = "Features Importance", figsize = (10,10))

xgbmodel = xgb.XGBRegressor(max_depth=5,
                            learning_rate=0.01,
                            n_estimators=10000,
                            objective='reg:linear',
                            gamma=1.45,
                            seed=random_seed,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.7,
                            colsample_bylevel=0.5)
Kfolder.validate(train, test, features, xgbmodel, name="xgbfinal", prepare_stacking=True)

catmodel = cat.CatBoostRegressor(iterations=10000,
                                 learning_rate=0.01,
                                 depth=5,
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200,
                                 random_seed=random_seed)
Kfolder.validate(train, test, features , catmodel, name="catfinal", prepare_stacking=True,
               fit_params={"use_best_model": True, "verbose": 100})

train['Revenue_lgb'] = train["lgbfinal"]

print("RMSE model lgb :" ,score(train, train.Revenue_lgb),)

train['Revenue_xgb'] = train["xgbfinal"]

print("RMSE model xgb :" ,score(train, train.Revenue_xgb))

train['Revenue_cat'] = train["catfinal"]

print("RMSE model cat :" ,score(train, train.Revenue_cat))

train['Revenue_Dragon1'] = 0.4 * train["lgbfinal"] + \
                               0.2 * train["xgbfinal"] + \
                               0.4 * train["catfinal"]

print("RMSE model Dragon1 :" ,score(train, train.Revenue_Dragon1))

train['Revenue_Dragon2'] = 0.35 * train["lgbfinal"] + \
                               0.3 * train["xgbfinal"] + \
                               0.35 * train["catfinal"]

print("RMSE model Dragon2 :" ,score(train, train.Revenue_Dragon2))

test['revenue'] =  np.expm1(test["lgbfinal"])
test[['id','revenue']].to_csv('submission_lgb.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1(test["xgbfinal"])
test[['id','revenue']].to_csv('submission_xgb.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1(test["catfinal"])
test[['id','revenue']].to_csv('submission_cat.csv', index=False)
test[['id','revenue']].head()


test['revenue'] =  np.expm1(0.4 * test["lgbfinal"]+ 0.4 * test["catfinal"] + 0.2 * test["xgbfinal"])
test[['id','revenue']].to_csv('submission_Dragon1.csv', index=False)
test[['id','revenue']].head()

test['revenue'] =  np.expm1((test["lgbfinal"] + test["catfinal"] + test["xgbfinal"])/3)
test[['id','revenue']].to_csv('submission_Dragon2.csv', index=False)
test[['id','revenue']].head()

# 1.77328
# 1.77534