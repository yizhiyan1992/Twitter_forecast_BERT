import pandas as pd
import numpy as np
import sklearn
from transformers import BertTokenizer


def transfer_to_tokens(text):
    maxlen=64
    tokenizer = BertTokenizer.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")
    input_ids=tokenizer.encode(text)
    print(text)
    input_ids=tokenizer.encode(text,max_length=maxlen)
    print(input_ids)
    print(tokenizer.convert_ids_to_tokens(input_ids))


def main():
    train = pd.read_csv('/Users/zhiyan1992/documents/github/tutorial/twitter_forecast_bert/data/train_processed.csv',index_col='id')
    test = pd.read_csv('/Users/zhiyan1992/documents/github/tutorial/twitter_forecast_bert/data/test_processed.csv',index_col='id')
    train=train[['text','target']]
    print(train.head())
    case1=train['text'].values[0]

    transfer_to_tokens(case1)

    #sklearn.model_selection.train_test_split

if __name__=="__main__":
    main()


