from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def preprocessing():
    genre_list = ['christian', "country-music", "hip-hop-rap", "pop", "rhythm-blues", "rock"]
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

    return x, y


x, y = preprocessing()

print(x[0])
def vectorizing(x):
    for i in range(len(x)):
        x[i] = str(x[i])

    vect = CountVectorizer(min_df=5).fit(x)  # 문자열을 토큰화, 어휘사전 구축  # 평균적으로 5일때 가장 정확도가 높음
    x = vect.transform(x)
    return x


x_vect = vectorizing(x)
x_train, x_test, y_train, y_test = train_test_split(x_vect, y, test_size=0.2, random_state=0)
all_pred = []
def train_metric(model):
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    all_pred.append(model_pred)
    ac_score = metrics.accuracy_score(y_test, model_pred)
    print("model : ",model.__class__.__name__)
    print("acc : ", ac_score)

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    score = cross_val_score(model, x_vect, y, cv=skf)
    print("stratified k-fold acc : ", score.mean())

    matrix = metrics.confusion_matrix(y_test,model_pred)

    print("\nconfusion matrix\n",matrix)

    label_score = []
    for i in range(6):
        label_score.append(round(matrix[i][i]/sum(matrix[i]),2))

    print("\n라벨 별 정답률 : ",label_score, "\n")


models = [
    MultinomialNB(fit_prior=True),
    RandomForestClassifier(n_estimators=100, n_jobs=-1),
    SVC(),
    MLPClassifier(max_iter=1000),
    GradientBoostingClassifier()
]

for m in models:
    train_metric(m)

total_pred = []

for i in range(106):
    label = [0,0,0,0,0,0]
    for j in range(5):
        label[all_pred[j][i]] += 1
    max_index = label.index(max(label))
    total_pred.append(max_index)

ac_score = metrics.accuracy_score(y_test, total_pred)
print("total acc : ", ac_score)