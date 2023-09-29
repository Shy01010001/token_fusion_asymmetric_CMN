# -*- coding: UTF-8 -*-
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from data import example_generator, abstract2sents


def compute_idf(data):
    dataset = []
    example = example_generator(data_path=config.train_data_path, single_pass=True)
    while True:
        try:
            e = example.next()
            try:
                if data == 'source':
                    article_text = e.features.feature['article'].bytes_list.value[0]
                    dataset.append(article_text)
                elif data == 'target':
                    abstract_tag = e.features.feature['abstract'].bytes_list.value[0]
                    abstract_text = ' '.join(abstract2sents(abstract_tag)) # string
                    dataset.append(abstract_text)
            except ValueError:
                continue
        except:
            break

    tf = TfidfVectorizer() # 类的一个实例
    tf.fit(dataset) # 类的一个方法
    keys = tf.get_feature_names()
    values = tf.idf_
    idfdict = dict(zip(keys, values))

    return idfdict


if __name__ == '__main__':
    src_idf_dict = compute_idf('source')
    trg_idf_dict = compute_idf('target')

    with open(os.path.join(config.idf_vocab_root, "src_Idf_Vocab.pickle"), 'wb') as handle:
        pickle.dump(src_idf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished writing idf vocabulary of source!")

    with open(os.path.join(config.idf_vocab_root, "trg_Idf_Vocab.pickle"), 'wb') as handle:
        pickle.dump(trg_idf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished writing idf vocabulary of target!")