
# coding: utf-8

# In[1]:


import os
import re
import pickle
import tarfile
import numpy as np

from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# In[2]:


BASE = os.path.abspath(os.path.curdir)


# In[3]:


MAXN = 12500


# In[4]:


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)


# - Imdb dataset url: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# In[5]:


def read_files(filetype):
    """
    filetype: 'train' or 'test'
    """
    all_labels = [1] * MAXN + [0] * MAXN
    
    pk_file = '%simdb_%s.pk' % (BASE, filetype)
    if os.path.exists(pk_file):
        all_texts = pickle.load(open(pk_file, 'rb'))
        return all_texts, all_labels
    
    
    all_texts = []
    file_list = []
    path = BASE + '/aclImdb/'
    
    # read positive
    pos_path = path + filetype + '/pos/'
    i = 0
    for f in os.listdir(pos_path):
        file_list.append(pos_path + f)
        i += 1
        if i == MAXN:
            break
    
    # read negative
    neg_path = path + filetype + '/neg/'
    i = 0
    for f in os.listdir(neg_path):
        file_list.append(neg_path + f)
        i += 1
        if i == MAXN:
            break
    
    for f in file_list:
        with open(f) as f:
            all_texts.append(rm_tags(' '.join(f.readlines())))
    
    # dump text
    
    with open(pk_file, 'wb') as f:
            pickle.dump(all_texts, f)
            
    return all_texts, all_labels     


# In[6]:


def preprocessing(train_texts, train_labels, test_texts, test_labels):
    token = Tokenizer(num_words=2000)
    token.fit_on_texts(train_texts)
    x_train_seq = token.texts_to_sequences(train_texts)
    x_test_seq = token.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=150, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test_seq, maxlen=150, padding='post', truncating='post')
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


# # Model TextCNN

# In[7]:


def text_cnn(maxlen=150, max_features=2000, embed_size=32):
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=1, activation='sigmoid')(output)

    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


# # Train

# In[8]:


if __name__ == '__main__':
    if not os.path.exists('./aclImdb'):
        tfile = tarfile.open(r'./aclImdb_v1.tar.gz', 'r:gz')  # r;gz是读取gzip压缩文件
        result = tfile.extractall('./')  # 解压缩文件到当前目录中
    train_texts, train_labels = read_files('train')
    test_texts, test_labels = read_files('test')
    x_train, y_train, x_test, y_test = preprocessing(train_texts, train_labels, test_texts, test_labels)
    
    print('train num : %s' % len(x_train))
    print('test num : %s' % len(x_test))
    
    model = text_cnn()
    batch_size = 128
    epochs = 20
    
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    
    predict = (np.asarray(model.predict(x_test))).round()
    f1 = f1_score(y_test, predict)
    acc = accuracy_score(y_test, predict)
    model.save(BASE + '/word_textcnn_acc_%.4f_f1_%.4f.h5' % (acc, f1))
    print('accuracy: %.4f, f1-score: %.4f' % (acc, f1))


# # 分析哪些影评被预测错误，是否存在极性转移或者Aspect-level Sentiment

# In[9]:


import pandas as pd


# In[10]:


# texts, labels = read_files('test')

# predict = pickle.load(open(BASE + '/predict.pk', 'rb'))

# len(labels) == len(predict)

# txt_labels = []
# txts = []
# for i in range(len(labels)):
#     if labels[i] != predict[i]:
#         txt_labels.append(labels[i])
#         txts.append(texts[i])

# len(txts) == 4013


# - 将这些写入文件好好看看

# In[11]:


# review_list = []
# for i in range(len(txts)):
#     js = {}
#     review = txts[i].replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
#     js['id'] = i
#     js['label'] = txt_labels[i]
#     js['content'] = review
#     review_list.append(js)


# df = pd.DataFrame(review_list)
# df = df[['label', 'content', 'id']]

# df.to_csv(BASE + '/hard_classify_imdb_review_data.csv', index=False, sep='\t')

