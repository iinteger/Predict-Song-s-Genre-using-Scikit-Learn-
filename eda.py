import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import Counter
import re


genre_list = ['christian', "country-music", "hip-hop-rap", "pop", "rhythm-blues", "rock"]


def preprocessing():
    # data load
    x = []
    y = []
    xAppend = x.append
    yAppend = y.append

    for i in range(len(genre_list)):
        song_list = os.listdir("./lyrics/"+genre_list[i])
        print(len(song_list))
        for j in range(len(song_list)):
            f = open("./lyrics/"+genre_list[i]+"/"+song_list[j], "r", encoding="UTF8")
            data = f.read().replace("\n", " ")  # 여러 줄로 나뉘어있는 가사를 한줄로 변환
            xAppend(data)
            yAppend(i)

    for i in range(len(x)):  # x 문장 토큰화
        x[i] = sent_tokenize(x[i])

    normalized_x = []
    for i in range(len(x)):  # 구두점 제거, 대문자를 소문자로 변환
        tokens = re.sub(r"[^a-z0-9]+", " ", str(x[i]).lower())
        normalized_x.append(tokens)

    x_tokenized = [word_tokenize(sentence) for sentence in normalized_x]  # x 단어 토큰화

    x = []
    stop_words = set(stopwords.words("english"))  # 불용어 제거
    for string in x_tokenized:
        not_stoped = []
        for word in string:
            if word not in stop_words:
                not_stoped.append(word)
        x.append(not_stoped)
    return x,y


x,y = preprocessing()


def w2v_visualising(x):  # 종교에 관한 워드가 많이 보임
    w2v = Word2Vec(sentences=x,min_count=2, workers=-1,)

    mpl.rcParams['axes.unicode_minus'] = False

    model = w2v
    vocab = list(model.wv.vocab)
    print(vocab[:20])
    X = model[vocab]

    tsne = TSNE(n_components=2)
    print(len(x))
    # 100개의 단어에 대해서만 시각화
    X_tsne = tsne.fit_transform(X[:100,:])
    df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])

    fig = plt.figure()
    fig.set_size_inches(40, 20)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])

    for word, pos in df.iterrows():
        ax.annotate(word, pos, fontsize=10)
    plt.show()
    return w2v


w2v = w2v_visualising(x)


def frequent_words(x):
    # 장르별 구분
    x_0 = x[:78]
    x_1 = x[78:164]
    x_2 = x[164:252]
    x_3 = x[252:350]
    x_4 = x[350:436]
    x_5 = x[436:]
    x_list = [x_0,x_1,x_2,x_3,x_4,x_5]

    vocab = Counter()  # 단어 빈도를 쉽게 세주는 모듈
    sentences = []
    n = 0
    for genre in x_list:
        vocab.clear()
        sentences.clear()
        for string in genre:
            result = []
            for word in string:
                if len(word) >2:
                    result.append(word)
                    vocab[word] +=+1
            sentences.append(result)
        print(genre_list[n],"의 빈도수 : ",vocab)
        n+=1

    vocab.clear()
    sentences.clear()
    # 장르 구분 없이 전체
    for string in x:
        result = []
        for word in string:
            if len(word) >2:
                result.append(word)
                vocab[word] +=+1
        sentences.append(result)
    print("총 단어 빈도수 : ",vocab)


frequent_words(x)


def kmeans_pca(x):
    vect = CountVectorizer()
    for i in range(len(x)):
        x[i] = str(x[i])

    x_vect = vect.fit_transform(x)

    kmeans = KMeans(n_clusters=3, init="random", random_state=0)  # init의 기본값인 k-means++는 문서 클러스터링에서 좋지 않다고 함
    kmeans.fit(x_vect)
    idx = list(kmeans.fit_predict(x))
    names = w2v.wv.index2word
    print(len(kmeans.labels_))

    DF = pd.DataFrame(data=x_vect)
    DF["target"] = y
    DF["cluster"]= kmeans.labels_
    print(kmeans.labels_)
    print(DF.groupby(['target','cluster'])[0].count())  # 그루핑 결과 특정 타겟으로 쏠림 현상이 나옴

    pca = SparsePCA(n_components=2, random_state=0, n_jobs=-1, verbose=1)  # 텍스트가 희소 행렬로 나타나기 때문에 sparse 메소드 사용
    #pca = PCA(n_components=2, random_state=0)
    pca_transformed = pca.fit_transform(x_vect.toarray())
    DF["pca_x"] = pca_transformed[:,0]
    DF["pca_y"] = pca_transformed[:,1]

    marker0_ind = DF[DF['cluster']==0].index
    marker1_ind = DF[DF['cluster']==1].index
    marker2_ind = DF[DF['cluster']==2].index

    plt.scatter(x=DF.loc[marker0_ind, 'pca_x'], y=DF.loc[marker0_ind,'pca_y'], marker='o')
    plt.scatter(x=DF.loc[marker1_ind, 'pca_x'], y=DF.loc[marker1_ind,'pca_y'], marker='v')
    plt.scatter(x=DF.loc[marker2_ind, 'pca_x'], y=DF.loc[marker2_ind,'pca_y'], marker='^')

    plt.xlabel("PCA1")
    plt.xlabel("PCA2")
    plt.show()


kmeans_pca(x)