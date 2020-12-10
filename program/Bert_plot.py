import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import Counter
train=pd.read_csv(r'/Users/Zhiyan1992/Desktop/train_processed.csv')
train_row=pd.read_csv(r'/Users/Zhiyan1992/Desktop/train.csv')
test_row=pd.read_csv(r'/Users/Zhiyan1992/Desktop/test.csv')

def yes_no_dis(train):
    label = train['target'].values
    Y = label[label == 1]
    N = label[label == 0]
    name_lis=['target=Yes','target=No']
    plt.bar([0,1],[Y.shape[0],N.shape[0]],tick_label=name_lis,color=['firebrick','midnightblue'])
    plt.title("Number of True and False labels in Training Set")
    plt.ylabel("Count")
    plt.xlabel("T/F target")
    plt.savefig('/Users/Zhiyan1992/Desktop/Fig1.png',dpi=300)
    plt.show()
    return

def key_words_distribution(train):
    C=Counter(train['keyword'])
    Count=[]
    for key in C.keys():
        Count.append([C[key],key])
    Count.sort(reverse=True)
    Count=Count[1:16]
    sns.barplot(x=[i[1] for i in Count],y=[i[0] for i in Count])
    plt.title('The count of the most 15 frequent keywords in training set')
    plt.xlabel('The top 15 keywords')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/Users/Zhiyan1992/Desktop/Fig2.png', dpi=300)
    plt.show()


def length(train,test):

    plt.subplot(1,2,1)
    plt.title("Text length distribution")
    plt.ylabel("Probability")
    text=train['text']
    train_now=[]
    for i in range(text.shape[0]):
        train_now.append(len(text[i]))
    plt.xlabel("(A) Train set")
    sns.distplot(train_now, bins=20,color='lightcoral')
    plt.subplot(1, 2, 2)
    print(train_now)
    test=test['text']
    test_now=[]
    for i in range(test.shape[0]):
        test_now.append(len(test[i]))
    sns.distplot(test_now, bins=20)
    plt.xlabel("(B) Test set")
    plt.show()

def country(train,test):
    C=Counter(test['location'])
    Count = []
    for key in C.keys():
        Count.append([C[key], key])
    Count.sort(reverse=True)
    Count = Count[1:16]
    print(Count)
    sns.barplot(x=[i[1] for i in Count],y=[i[0] for i in Count])
    plt.title('The count of the most 15 frequent locations in test set')
    plt.xlabel('The top 15 keywords')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/Users/Zhiyan1992/Desktop/Fig4_b.png', dpi=300)
    plt.show()

#yes_no_dis(train)
#key_words_distribution(train)
#length(train_row,test_row)
country(train_row,test_row)