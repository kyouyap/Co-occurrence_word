import tweepy

# 認証キーの設定
consumer_key = "hoge"
consumer_secret = "hoge"
access_token = "hoge"
access_token_secret = "hoge"

# OAuth認証
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# APIのインスタンスを生成
# api = tweepy.API(auth)
api = tweepy.API(
    auth,
    wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True
)

def getFollowers_ids(api, screen_name):
    followers_ids = tweepy.Cursor(api.followers_ids, screen_name=screen_name, cursor=-1).items()
    followers_ids_list = []
    try:
        followers_ids_list=[i for i in followers_ids]
        

    except tweepy.error.TweepError as e:
        print(e.reason)

    return followers_ids_list
    

# Get Id list of followers
from tqdm import tqdm
import time
screen_name="kyouyap"
followers_ids=getFollowers_ids(api, screen_name)
name=[]
screen_name=[]
url=[]
description=[]
protected=[]
folloerws_count=[]
friends＿count=[]
listed_count=[]
statuses_count=[]
created_at=[]
try:
    for i in tqdm(range(len(followers_ids))):
        if (i%900==0)and(i!=0):
            time.sleep(600)
        center_info = api.get_user(id=followers_ids[i])
        name.append(center_info.name)
        screen_name.append(center_info.screen_name)
        url.append(center_info.url)
        description.append(center_info.description)
        protected.append(center_info.protected)
        folloerws_count.append(center_info.followers_count)
        friends＿count.append(center_info.friends_count)
        listed_count.append(center_info.listed_count)
        statuses_count.append(center_info.statuses_count)
        created_at.append(center_info.created_at)
        if (i%900==0)and(i!=0):
            time.sleep(900)
except tweepy.error.TweepError as e:
    print(e.reason)
import pandas as pd
df=pd.DataFrame({"name":name,"screen_name":screen_name,"url":url
                 ,"description":description,"protected":protected,"folloerws_count":folloerws_count,"friends_count":friends＿count,"listed_count":listed_count,"statuses_count":statuses_count,"created_at":created_at})
df.to_csv("folower_information.csv",index=None,encoding='utf_8_sig')

df.description=df.description.fillna("")

import japanize_matplotlib
import re
import MeCab
import numpy as np

# 対象の品詞
TARGET_POS1 = ['名詞']
 
# 対象の詳細分類1
TARGET_POS2 = ['サ変接続', 'ナイ形容詞語幹', '形容動詞語幹', '一般', '固有名詞']
 
# ストップワード
STOP_WORDS = ['*']
 
def remove_blank(chapter):
    # 空白行と段落先頭の空白を削除
 
    lines = chapter.splitlines()
 
    # 空白行削除
    # 行頭の空白削除
    lines_cleaned = [l.strip() for l in lines if len(l) != 0]
 
    return '\n'.join(lines_cleaned)
 
def chapter2bform(chapter_l):
    # 章ごとに形態素解析して単語の原型のリストを作成
 
    m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')
 
    bform_2l = []
    for i, chapter in enumerate(chapter_l):
        node = m.parseToNode(chapter)
 
        bform_l = []
        while node:
            feature_split = node.feature.split(',')
 
            pos1 = feature_split[0]
            pos2 = feature_split[1]
            base_form = feature_split[6]
 
            if pos1 in TARGET_POS1 and pos2 in TARGET_POS2 and base_form not in STOP_WORDS:
                bform_l.append(base_form)
 
            node = node.next
 
        bform_2l.append(bform_l)
 
        print('Term number of chapter {}: '.format(i+1), len(bform_l))
    return bform_2l

from itertools import combinations, dropwhile
from collections import Counter, OrderedDict
 
def bform2pair(bform_2l, min_cnt=5):
    # 単語ペアの出現章数をカウント
 
    # 全単語ペアのリスト
    pair_all = []
 
    for bform_l in bform_2l:
        # 章ごとに単語ペアを作成
        # combinationsを使うと順番が違うだけのペアは重複しない
        # ただし、同単語のペアは存在しえるのでsetでユニークにする
        pair_l = list(combinations(set(bform_l), 2))
 
        # 単語ペアの順番をソート
        for i,pair in enumerate(pair_l):
            pair_l[i] = tuple(sorted(pair))
 
        pair_all += pair_l
 
    # 単語ペアごとの出現章数
    pair_count = Counter(pair_all)
 
    # ペア数がmin_cnt以上に限定
    for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
        del pair_count[key]
 
    return pair_count
 
def pair2jaccard(pair_count, bform_2l, edge_th=0.4):
    # jaccard係数を計算
 
    # 単語ごとの出現章数
    word_count = Counter()
    for bform_l in bform_2l:
        word_count += Counter(set(bform_l))
 
    # 単語ペアごとのjaccard係数を計算
    jaccard_coef = []
    for pair, cnt in pair_count.items():
        jaccard_coef.append(cnt / (word_count[pair[0]] + word_count[pair[1]] - cnt))
 
    # jaccard係数がedge_th未満の単語ペアを除外
    jaccard_dict = OrderedDict()
    for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
        if coef >= edge_th:
            jaccard_dict[pair] = coef
            print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')
 
    return jaccard_dict
 

import networkx as nx
# matplotlibのターミナル対応
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
def build_network(jaccard_dict):
    # 共起ネットワークを作成
    G = nx.Graph()
 
    #  接点／単語（node）の追加
    # ソートしないとネットワーク図の配置が実行ごとに変わる
    nodes = sorted(set([j for pair in jaccard_dict.keys() for j in pair]))
    G.add_nodes_from(nodes)
 
    print('Number of nodes =', G.number_of_nodes())
 
    #  線（edge）の追加
    for pair, coef in jaccard_dict.items():
        G.add_edge(pair[0], pair[1], weight=coef)
 
    print('Number of edges =', G.number_of_edges())
 
    plt.figure(figsize=(15, 15))
 
    # nodeの配置方法の指定
    seed = 0
    np.random.seed(seed)
    pos = nx.nx_agraph.graphviz_layout(
    G,
    prog='neato',
    args='-Goverlap="scalexy" -Gsep="+6" -Gnodesep=0.8 -Gsplines="polyline" -GpackMode="graph" -Gstart={}'.format(seed))

 
    # nodeの大きさと色をページランクアルゴリズムによる重要度により変える
    pr = nx.pagerank(G)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=list(pr.values()),
        cmap=plt.cm.rainbow,
        alpha=0.7,
        node_size=[100000*v for v in pr.values()])
 
    # 日本語ラベルの設定
    nx.draw_networkx_labels(G, pos, fontsize=15, font_family='IPAexGothic', font_weight='bold')
 
    # エッジ太さをJaccard係数により変える
    edge_width = [d['weight'] * 8 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='darkgrey', width=edge_width, font_size=12, font_family='IPAexGothic')
 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('co-occurance.png', bbox_inches='tight')



# 章ごとの単語原型リスト
bform_2l = chapter2bform(list(df.description))

# Jaccard係数の計算
pair_count = bform2pair(bform_2l, min_cnt=4)
jaccard_dict = pair2jaccard(pair_count, bform_2l, edge_th=0.4)

# 共起ネットワーク作成
build_network(jaccard_dict)