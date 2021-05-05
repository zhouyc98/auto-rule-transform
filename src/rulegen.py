from ruleparse import *
import ruleparse
from owlready2 import *
import json
import jieba_fast as jieba
from gensim.models import Word2Vec
from scipy import spatial
from gensim import models
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from collections import Counter
import math
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore", category=Warning)

CMP_DICT_Onto = OrderedDict([('<=', '小于等于 小于或等于 不大于 不高于 不多于 不超过'.split()),
                             ('>=', '大于等于 大于或等于 不小于 不低于 不少于'.split()),
                             ('>', '大于 超过 高于'.split()),
                             ('<', '小于 低于'.split()),
                             ('!=', '不等于 避免 不采用 无法采用'.split()),
                             ('=', '等于 为 采用 用 按照 按 符合 执行'.split()),
                             ('has no', '不有 不设置 不设 不具备 无'.split()),
                             ('has', '有 具有 含有 留有 设 设置 设有 增设 铺设 搭设 安装 具备'.split()),
                             ('not in', '不在'.split()),
                             ('in', '在'.split()),
                             ])
DEONTIC_WORDS = ('应当', '应该', '应按', '应能', '尚应', '应', '必须', '尽量', '要', '宜', '得')  # '得' 通常在 '不得' 中使用

'''
State: Use
Funtion: Get compare operation by text
'''


def get_cmp_str_onto(cmp_):
    cmp_value = cmp_.values if isinstance(cmp_, RCNode) else cmp_

    if not cmp_value:  # None or ''
        return DEFAULT_TAG_VALUES['cmp']

    for dw in DEONTIC_WORDS:
        cmp_value = cmp_value.replace(dw, '')

    # simplify cmp_value, if can
    if cmp_value == '':
        return '='

    for key, values in CMP_DICT_Onto.items():
        if cmp_value in values:
            return key

    return cmp_value


'''
State: discard
Function: This class reads the predefined class in the excel table by key words mapping, it's a test function, but it's hard to maintain the excel.
'''


class Keywords_dict():
    def __init__(self, path='./data/OntoKeywords.xlsx'):
        df = pd.read_excel(path, sheet_name=['Class', 'DataProperty', 'ObjectProperty', 'CommonExpression'],
                           header=None)
        # get class dict
        class_dict_data = df['Class']
        self.class_dict = {}
        for i in class_dict_data.index.values:
            row_data = class_dict_data.values[i, :]
            class_name = row_data[0]
            if class_name not in self.class_dict:
                self.class_dict[class_name] = []
                for class_keywords in row_data[1:]:
                    if type(class_keywords) == str:
                        self.class_dict[class_name].append(class_keywords)

        # get dataproperty dict
        dataproperty_dict_data = df['DataProperty']
        self.dataproperty_dict = {}
        for i in dataproperty_dict_data.index.values:
            row_data = dataproperty_dict_data.values[i, :]
            dataproperty_name = row_data[0]
            if dataproperty_name not in self.dataproperty_dict:
                self.dataproperty_dict[dataproperty_name] = []
                for dataproperty_keywords in row_data[1:]:
                    if type(dataproperty_keywords) == str:
                        self.dataproperty_dict[dataproperty_name].append(dataproperty_keywords)

    def get_OntoClass_Name(self, phrase: str):
        '''可以通过文本相似度的方式进行计算+维特比算法，暂时先使用字符串匹配'''
        for class_name in self.class_dict:
            if phrase in self.class_dict[class_name]:
                return ['class', class_name]
        for dataproperty_name in self.dataproperty_dict:
            if phrase in self.dataproperty_dict[dataproperty_name]:
                return ['dataproperty', dataproperty_name]


'''
State: use
Function: get the split sentence with no stopwords
'''


class Sentence:
    def __init__(self, sentence, stopwords):
        self.raw = sentence
        self.tokens = [t for t in jieba.cut(sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in stopwords]

    def tokens_no_stop(self):
        return self.tokens_without_stop

    def tokens_(self):
        return self.tokens

    def tfidf_weight(self, dictionary, tfidf_model):
        corpus = [dictionary.doc2bow(self.tokens_without_stop)]
        token_tf_tdf = list(tfidf_model[corpus])[0]
        word_tf_tdf = [(dictionary[word_pair[0]], word_pair[1]) for word_pair in token_tf_tdf]
        return word_tf_tdf


'''
State: Use
Function: Read the class and dataproperty and their description in the ontology (.owl file), and then save them into dict (.pkl file)
DataStructure: dict
    ontology_pkl [ontology_class, ontology_dataproperty, ontology_objectproperty]
    ontology_class [class1, class2, ...]
    ontology_dataproperty [dataproperty1, dataproperty2, ...]
    ontology_objectproperty [ontology_objectproperty1, ontology_objectproperty2, ...]
    class1 [className, classURL, comments [comment1, commentt2, ...]]
    dataproperty1 [datapropertyName, datapropertyURL, comments [comment1, commentt2, ...]]
    objectproperty1 [objectpropertyName, objectpropertyURL, comments [comment1, commentt2, ...]]
'''


def onto_info_extract(src=r"..\data\ontology\BuildingDesignFireCodesOntology.owl",
                      tag=r'..\data\ontology\BuildingDesignFireCodesOntology.pkl'):
    onto = get_ontology(src).load()
    ontology_pkl = []
    ontology_class = []
    ontology_dataproperty = []
    ontology_objectproperty = []
    for oneclass in onto.classes():
        ontology_class.append([str(oneclass.name), str(oneclass), list(oneclass.comment)])
    for onedataproperty in onto.data_properties():
        ontology_dataproperty.append([str(onedataproperty.name), str(onedataproperty), list(onedataproperty.comment)])
    for oneobjproperty in onto.object_properties():
        ontology_objectproperty.append([str(oneobjproperty.name), str(oneobjproperty), list(oneobjproperty.comment)])
    ontology_pkl.append(ontology_class)
    ontology_pkl.append(ontology_dataproperty)
    ontology_pkl.append(ontology_objectproperty)
    with open(tag, 'wb') as fout:
        pickle.dump(ontology_pkl, fout)
    '''Bug: Object of type "xxx" is not JSON serializable occur when using json'''
    # with open(tag, 'w', encoding='utf-8') as fout:
    #     jstr = json.dumps(ontology_pkl, ensure_ascii=False)
    #     fout.write(jstr)


'''
State: use
Function: get TF-IDF from rules, and save the dict
'''


def rules_TFIDF(corpus_path=r'..\data\rules\allRules.text'):
    stoplist = stopwordslist(r'.\models\word2vec\Stopwords.txt')
    dictionary_path = r'.\models\tfidf\rules_doc2bow.dict'
    tfidf_model_path = r'.\models\tfidf\rules_tfidf.model'

    with open(corpus_path, 'r', encoding='UTF-8') as f1:
        corpus = f1.readlines()
        texts = [[word for word in document.split(' ') if word not in stoplist] for document in corpus]

    dictionary = corpora.Dictionary(texts)
    print('Dictionary:', dictionary.token2id)
    print('*' * 10)
    dictionary.save(dictionary_path)
    print('Now caculating tf-idf')
    tfidf_corpus = [dictionary.doc2bow(text) for text in texts]
    tf_idf_model = TfidfModel(tfidf_corpus, normalize=False)
    tf_idf_model.save(tfidf_model_path)
    # word_tf_tdf = list(tf_idf_model[tfidf_corpus])
    # print('词频:', tfidf_corpus)
    # print('词的tf-idf值:', word_tf_tdf)


'''
State: use
Function: merge the label_config.json and FireCode_labeled.jsonl file exported by docanno for better similarity matching use
'''


def process_data_docanno(
        src_config=r'..\data\docanno\label_config.json',
        src_labels=r'..\data\docanno\FireCode_labeled.jsonl',
        tag='..\data\docanno\FireCode_label_merge.json'):
    sentences = []
    labels = []
    sentences_labels = []
    with open(src_labels, 'r', encoding='utf-8') as f1, open(src_config, 'r', encoding='utf-8') as f2:
        rawdata = f1.readlines()
        for oneline in rawdata:
            sentences.append(json.loads(oneline))
        labels = json.load(f2)
    for onesentence in sentences:
        id = onesentence['id']
        text = onesentence['text']
        annotations = onesentence['annotations']
        annotations_merge = []
        for oneannotation in annotations:
            word_label = labels[int(oneannotation['label']) - 1]['text']
            start_offset = int(oneannotation['start_offset'])
            end_offset = int(oneannotation['end_offset'])
            word = text[start_offset:end_offset]
            annotations_merge.append({"label": word_label, "word": word})
        sentences_labels.append({'id': id, 'text': text, 'annotations': annotations_merge})
    with open(tag, 'w', encoding='utf-8') as fout:
        jstr = json.dumps(sentences_labels, ensure_ascii=False)
        replaces = [('}, ', '\n},\n'), ('[{', '[\n{'), ('}]', '\n}\n]'), ('", "', '",\n"'), ('], "', '],\n"'),
                    ('{"', '{\n"'), ('"id', '    "id'), ('"text', '    "text'), ('"annotations', '    "annotations'),
                    ('"label', '    "label'), ('"word', '    "word')]
        for r1, r2 in replaces:
            jstr = jstr.replace(r1, r2)
        fout.write(jstr)
    print(f'The two docanno file have been merged in: {tag}')


def stopwordslist(stopWordsFile):
    stopwords = [line.strip() for line in open(stopWordsFile, encoding='UTF-8').readlines()]
    return stopwords


def is_stop_words(word: str, stopwords):
    return word in stopwords


'''
State: use
Function: give a ontology word and the corresponding set of description, calculate the similarity between a natural language word and the ontology word based
Method: 
    1. This method calculate the similartiy based on each word, and then mean similarity
    2. This method calculate the vector then mean vector, and calculate the similarity
    3. Take the tf-idf of each word as the weight, and the word similarity of all words is weighted averaged
    4. Word mover's distance, The Word Mover's Distance uses the word embeddings of the words in two texts to measure 
       the minimum amount that the words in one text need to "travel" in semantic space to reach the words of the other text.
       Word mover's distance is available in the popular Gensim library.
    5. To compute SIF sentence embeddings, we first compute a weighted average of the token embeddings in the sentence. 
       This procedure is very similar to the weighted average we used above, with the single difference that the word embeddings are weighted by a/a+p(w), 
       where a is a parameter that is set to 0.001 by default, and p(w) is the estimated relative frequency of a word in a reference corpus.
       Next, we need to perform common component removal: we compute the principal component of the sentence embeddings,
       we obtained above and subtract from them their projections on this first principal component. 
       This corrects for the influence of high-frequency words that mostly have a syntactic or discourse function
       
Inout: 
    onto_description: list(str)
    word: str
    word2vec_model: pretrained word2vec word2vec_model
    stopword: stopwords list
output:
    similarity: float
Reference:
    https://github.com/nlptown/nlp-notebooks/blob/master/Simple%20Sentence%20Similarity.ipynb
'''


def word2vec_similarity(onto_description, word, word2vec_model, stopwords, dictionary, method=1):
    if len(onto_description) == 0 and method != 4 and method != 5:
        return 0
    elif len(onto_description) == 0 and method == 4:
        return -1000
    elif len(onto_description) == 0 and method == 5:
        return -10

    if method == 1:
        '''
        method = 1 This method calculate the similartiy then mean similarity
        '''
        input_word_seq = Sentence(word, stopwords).tokens_no_stop()
        caculation_times = 0
        similarity = 0
        for one_onto_description in onto_description:
            description_seq = Sentence(one_onto_description, stopwords).tokens_no_stop()
            for word_seq_item in input_word_seq:
                for description_seq_item in description_seq:
                    # Eliminate the influence of meaningless words
                    if word_seq_item in word2vec_model and description_seq_item in word2vec_model:
                        similarity += word2vec_model.wv.similarity(word_seq_item, description_seq_item)
                        caculation_times += 1
        return similarity / caculation_times
    elif method == 2:
        def getVector(cutWords, word2vec_model):
            '''diffierent similarity method: https://zhuanlan.zhihu.com/p/108127410'''
            '''
            State: Use
            Function: give a list of word, caculated the average vector of it
            '''
            i = 0
            # index2word_set = set(word2vec_model.wv.index2word)
            article_vector = np.zeros((word2vec_model.layer1_size))
            for cutWord in cutWords:
                if cutWord in word2vec_model:
                    article_vector = np.add(article_vector, word2vec_model.wv[cutWord])
                    i += 1
            cutWord_vector = np.divide(article_vector, i)
            return cutWord_vector

        '''
        method = 2 This method calculate the vector then mean vector, and calculate the similarity
        '''
        input_word_seq = Sentence(word, stopwords).tokens_no_stop()
        caculation_times = 0
        similarity = 0
        input_word_vec = getVector(input_word_seq, word2vec_model)
        description_vecs = []
        for one_onto_description in onto_description:
            description_seq = Sentence(one_onto_description, stopwords).tokens_no_stop()
            description_vecs.append(getVector(description_seq, word2vec_model))
        for description_vec in description_vecs:
            caculation_times += 1
            similarity += 1 - spatial.distance.cosine(input_word_vec, description_vec)
        return similarity / caculation_times

    elif method == 3:
        '''
            method = 3 This method calculate the vector then mean vector,Take the tf-idf of each word as the weight, 
            and the word similarity of all words is weighted averaged and calculate the similarity
            Reference: https://blog.csdn.net/laojie4124/article/details/90378868
            Reference: https://github.com/nlptown/nlp-notebooks/blob/master/Simple%20Sentence%20Similarity.ipynb
        '''

        def tfidf_weigh_similarity(sentence1, sentence2, word2vec_model, dictionary, use_stoplist=True):
            if dictionary is not None:
                N = dictionary.num_docs
            sims = []
            for (sent1, sent2) in zip(sentence1, sentence2):
                tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
                tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

                tokens1 = [token for token in tokens1 if token in word2vec_model]
                tokens2 = [token for token in tokens2 if token in word2vec_model]

                if len(tokens1) == 0 or len(tokens2) == 0:
                    sims.append(0)
                    continue

                tokfreqs1 = Counter(tokens1)
                tokfreqs2 = Counter(tokens2)

                weights1 = []
                weights2 = []
                for token1 in tokfreqs1:
                    if token1 in dictionary.token2id:
                        weights1.append(
                            tokfreqs1[token1] * math.log(N / (dictionary.dfs[dictionary.token2id[token1]] + 1)))
                    else:
                        weights1.append(
                            tokfreqs1[token1] * math.log(N / (N - 1)))
                for token2 in tokfreqs2:
                    if token2 in dictionary.token2id:
                        weights2.append(
                            tokfreqs2[token2] * math.log(N / (dictionary.dfs[dictionary.token2id[token2]] + 1)))
                    else:
                        weights2.append(
                            tokfreqs2[token2] * math.log(N / (N - 1)))

                embedding1 = np.average([word2vec_model.wv[token] for token in tokfreqs1], axis=0,
                                        weights=weights1).reshape(1,
                                                                  -1)
                embedding2 = np.average([word2vec_model.wv[token] for token in tokfreqs2], axis=0,
                                        weights=weights2).reshape(1,
                                                                  -1)

                sim = 1 - spatial.distance.cosine(embedding1, embedding2)
                sims.append(sim)
            return sims

        description_sentences = []
        input_sentences = []
        for one_onto_description in onto_description:
            description_sentences.append(Sentence(one_onto_description, stopwords))
            input_sentences.append(Sentence(word, stopwords))
        sims = tfidf_weigh_similarity(description_sentences, input_sentences, word2vec_model, dictionary)
        return np.mean(sims)
    elif method == 4:
        def wmd_similarity(sentences1, sentences2, word2vec_model):
            sims = []
            for (sent1, sent2) in zip(sentences1, sentences2):
                tokens1 = sent1.tokens_without_stop
                tokens2 = sent2.tokens_without_stop

                tokens1 = [token for token in tokens1 if token in word2vec_model]
                tokens2 = [token for token in tokens2 if token in word2vec_model]

                if len(tokens1) == 0 or len(tokens2) == 0:
                    tokens1 = [token for token in sent1.tokens if token in word2vec_model]
                    tokens2 = [token for token in sent2.tokens if token in word2vec_model]
                sims.append(-word2vec_model.wmdistance(tokens1, tokens2))
            return np.mean(sims)

        description_sentences = []
        input_sentences = []
        for one_onto_description in onto_description:
            description_sentences.append(Sentence(one_onto_description, stopwords))
            input_sentences.append(Sentence(word, stopwords))
        sims = wmd_similarity(description_sentences, input_sentences, word2vec_model)
        similarity = np.mean(sims)
        return similarity
    elif method == 5:
        def remove_first_principal_component(X):
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(X)
            pc = svd.components_
            XX = X - X.dot(pc.transpose()) * pc
            return XX

        def sif_similarity(sentences1, sentences2, word2vec_model, dictionary, a=0.001):
            total_freq = dictionary.num_pos

            embeddings = []
            # SIF requires us to first collect all sentence embeddings and then perform
            # common component analysis.
            for (sent1, sent2) in zip(sentences1, sentences2):
                tokens1 = sent1.tokens_without_stop
                tokens2 = sent2.tokens_without_stop

                tokens1 = [token for token in tokens1 if token in word2vec_model]
                tokens2 = [token for token in tokens2 if token in word2vec_model]

                weights1 = []
                weights2 = []
                for token1 in tokens1:
                    if token1 in dictionary.token2id:
                        weights1. append(a / (a + dictionary.cfs[dictionary.token2id[token1]] / total_freq))
                    else:
                        weights1.append(a / (a + 1000 / total_freq))
                for token2 in tokens2:
                    if token2 in dictionary.token2id:
                        weights2. append(a / (a + dictionary.cfs[dictionary.token2id[token2]] / total_freq))
                    else:
                        weights2.append(a / (a + 1000 / total_freq))

                embedding1 = np.average([word2vec_model[token] for token in tokens1], axis=0, weights=weights1)
                embedding2 = np.average([word2vec_model[token] for token in tokens2], axis=0, weights=weights2)

                embeddings.append(embedding1)
                embeddings.append(embedding2)

            embeddings = remove_first_principal_component(np.array(embeddings))
            sims = [1 - spatial.distance.cosine(embeddings[idx * 2].reshape(1, -1),
                                                embeddings[idx * 2 + 1].reshape(1, -1))
                    for idx in range(int(len(embeddings) / 2))]

            return sims

        description_sentences = []
        input_sentences = []
        for one_onto_description in onto_description:
            description_sentences.append(Sentence(one_onto_description, stopwords))
            input_sentences.append(Sentence(word, stopwords))
        sims = sif_similarity(description_sentences, input_sentences, word2vec_model, dictionary)
        similarity = np.mean(sims)
        return similarity


'''
State: use
Function: give a word list, for each word find the most similar term in ontology
Input:
    word: str
    ontology: pickle file
output:
    onto_term: str
'''


def most_similar_onto_term(words, method=1):
    jieba.load_userdict(r'.\models\word2vec\wordsList500.txt')
    stopwords = stopwordslist(r'.\models\word2vec\Stopwords.txt')
    model = Word2Vec.load(r'.\models\word2vec\Merge.model')
    onto_file = r'..\data\ontology\BuildingDesignFireCodesOntology.pkl'
    dictionary = corpora.Dictionary.load(r"./models/tfidf/rules_doc2bow.dict")

    with open(onto_file, 'rb') as f:
        onto_list = pickle.load(f)
    ontology_class = onto_list[0]
    ontology_dataproperty = onto_list[1]
    ontology_objectproperty = onto_list[2]
    term_word_pair = []
    for word in words:
        similarity_scores = {}
        for one_class in ontology_class:
            class_name = one_class[0]
            class_description = one_class[2]
            similarity_scores[class_name] = word2vec_similarity(class_description, word, model, stopwords, dictionary,
                                                                method=method)
        for one_dataproperty in ontology_dataproperty:
            dataproperty_name = one_dataproperty[0]
            dataproperty_description = one_dataproperty[2]
            similarity_scores[dataproperty_name] = word2vec_similarity(dataproperty_description, word, model,
                                                                       stopwords, dictionary, method=method)
        similarity_scores_sort = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
        for key in similarity_scores_sort:
            onto_term = key
            term_word_pair.append({"label": onto_term, "word": word})
            break
    return term_word_pair


'''
State: Use
Function: test for the most_similar_onto_term using the doccano tag words， metric using acc first
'''


def __test_for_el(docanno_src='./data/FireCode_label_merge.json', method=1):
    with open(docanno_src, 'r', encoding='utf-8') as f1:
        test_data = json.load(f1)
    annotations = []
    for sentence in test_data:
        for annotation in sentence['annotations']:
            annotations.append(annotation)
    words = []
    for annotation in annotations:
        words.append(annotation['word'])
    predicitons = most_similar_onto_term(words, method=method)
    assert len(predicitons) == len(annotations), 'Somewords in annotations test dataset is not predict'
    true_num = 0
    for index in range(len(annotations)):
        assert predicitons[index]['word'] == annotations[index][
            'word'], 'the prediction word is not the same as that in test dataset'
        # data structure:
        #   predictions {'label':(term,similarity_score), 'word':word}
        #   annotaions {'label':term, 'word':word}
        if predicitons[index]['label'][0].lower() == annotations[index]['label'].lower():
            true_num += 1
        else:
            print('*' * 10)
            print(f'Word: {annotations[index]["word"]}')
            print(f'True: {annotations[index]["label"]}')
            print(f'Predict: {predicitons[index]["label"]}')

    acc = true_num / len(predicitons)
    print(f'The acc of the word2vec method is {acc} %')


def RCNode_ontoclass():
    RCtrees = ruleparse.pkl_data_loader()
    ontoclass_dict = Keywords_dict()

    def preorder(RCtree, ontoclass_dict):
        words = RCtree.curr_node.values
        if words is not None:
            class_type, class_name = ontoclass_dict.get_OntoClass_Name(words)
            RCtree.curr_node.add_ontoclass(class_type, class_name)
        while RCtree.curr_node.has_child():
            for one_childnode in RCtree.curr_node.child_nodes:
                RCtree.curr_node = one_childnode
                preorder(RCtree, ontoclass_dict)

    for one_RCtree in RCtrees:
        preorder(one_RCtree, ontoclass_dict)
        one_RCtree.curr_node = one_RCtree.root
    return RCtrees


class RCtriple():
    def __init__(self, RCtree):
        self.RCtree = RCtree

    def generate_sparql(self):
        '''先全部遍历，定义一遍class， 再全部遍历一遍class的二元关系'''

        def preorder_classdefine(RCtree, sparql_con):
            if hasattr(RCtree.curr_node, 'ontoclass_type'):
                # to add sparql pronoun for class node, for example ?element
                if not hasattr(RCtree.curr_node, 'sparql_pronoun'):
                    pronoun_count = RCtree.count_node_pronoun()
                    RCtree.curr_node.add_sparql_pronoun(pronoun_count)

                if RCtree.curr_node.ontoclass_type == 'class':
                    '''Now only deal with the most simple case, one element'''
                    sparql_pronoun = RCtree.curr_node.sparql_pronoun
                    sparql_con += sparql_pronoun + ' rdf:type owl:NamedIndividual , myclass:' + RCtree.curr_node.ontoclass_name + ' .\n\t'
                    sparql_con += sparql_pronoun + ' :hasGlobalId ' + sparql_pronoun + '_id .\n\t'

            while RCtree.curr_node.has_child():
                for one_childnode in RCtree.curr_node.child_nodes:
                    RCtree.curr_node = one_childnode
                    # need to return value otherwise it will return none
                    return preorder_classdefine(RCtree, sparql_con)
            return sparql_con

        def preorder_relation(RCtree, sparql_con):
            if hasattr(RCtree.curr_node, 'ontoclass_type'):
                assert hasattr(RCtree.curr_node, 'sparql_pronoun'), 'The sparql class pronoun is not gen complete yet!'
                if RCtree.curr_node.ontoclass_type == 'class':
                    # only do it when the current node type = class
                    for one_childnode in RCtree.curr_node.child_nodes:
                        if hasattr(one_childnode, 'ontoclass_type'):
                            if one_childnode.ontoclass_type == 'dataproperty':
                                # to add sparl pronoun for dataproperty node, for example ?dataproperty
                                if not hasattr(one_childnode, 'sparql_pronoun'):
                                    pronoun_count = RCtree.count_node_pronoun()
                                    one_childnode.add_sparql_pronoun(pronoun_count)
                                req_cmp_rawdata = one_childnode.req[0].values
                                req_cmp = get_cmp_str_onto(req_cmp_rawdata)
                                req_value_rawdata = one_childnode.req[1].values
                                req_value = re.findall(r"\d+\.?\d*", req_value_rawdata)[0]

                                sparql_con += RCtree.curr_node.sparql_pronoun + ' :' + one_childnode.ontoclass_name + ' ' + one_childnode.sparql_pronoun + ' .\n\t'
                                sparql_con += 'BIND ((' + one_childnode.sparql_pronoun + ' ' + req_cmp + ' \'' + req_value + '\'^^xsd:decimal) AS ?Pass' + ')\n\t'  # 'BIND ((?dataproperty1 >= '2.0'^^xsd:decimal) AS ?Pass)'
                                sparql_con += 'FILTER (?Pass' + " = 'false'^^xsd:boolean)\n\t"  # FILTER (?Pass = 'false'^^xsd:boolean)
                            elif one_childnode.ontoclass_type == 'class':
                                pass

            while RCtree.curr_node.has_child():
                for one_childnode in RCtree.curr_node.child_nodes:
                    RCtree.curr_node = one_childnode
                    return preorder_relation(RCtree, sparql_con)
            return sparql_con

        sparql_con = ''
        sparql_con = preorder_classdefine(self.RCtree, sparql_con=sparql_con)
        self.RCtree.curr_node = self.RCtree.root
        sparql_con = preorder_relation(self.RCtree, sparql_con=sparql_con)
        sparql_prefix = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nPREFIX myclass: <http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#>\nPREFIX : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#>\nSELECT DISTINCT *\nWHERE {\n\t'
        sparql_full = sparql_prefix + sparql_con + '}\n'
        return sparql_full


if __name__ == '__main__':
    '''
        old version of the automated sparql generated code
    '''
    # RCtrees= RCNode_ontoclass()
    # aRCtriple = RCtriple(RCtrees[0])
    # sparql_full= aRCtriple.generate_sparql()
    # print(sparql_full)

    '''
        BuildingDesignFireCodesOntology.pkl file generator, when the .owl file changes, this function should run again
    '''
    # src = r"..\data\ontology\BuildingDesignFireCodesOntology.owl"
    # tag = r'..\data\ontology\BuildingDesignFireCodesOntology.pkl'
    # onto_info_extract(src=src, tag=tag)

    '''
        Change docanno annotation file and label file into one file 
    '''
    # src_config = r'..\data\docanno\label_config.json'
    # src_labels = r'..\data\docanno\FireCode_labeled.jsonl'
    # tag_path = r'..\data\docanno\FireCode_label_merge.json'
    # process_data_docanno(src_config=src_config, src_labels=src_labels, tag=tag_path)

    '''
        test for most_similar_onto_term
    '''
    # print(most_similar_onto_term(['耐火极限', '耐火等级']))

    '''
        test for __test_for_el
    '''
    docanno_src = '..\data\docanno\FireCode_label_merge.json'
    __test_for_el(docanno_src=docanno_src, method=5)

    '''
        pre-train for tf-idf
    '''
    # rules_TFIDF()
    # dictionary = corpora.Dictionary.load(r"./models/tfidf/rules_doc2bow.dict")
    # model_load = models.TfidfModel.load(r"./models/tfidf/rules_tfidf.word2vec_model")
    # stoplist = stopwordslist(r'.\models\word2vec\Stopwords.txt')
    # sentence1 = Sentence('生产的火灾危险性类别为乙级，厂房的耐火等级为二级的高层厂房，面积不超过1500m2。', stoplist)
    # print(sentence1.tfidf_weight(dictionary, model_load))
