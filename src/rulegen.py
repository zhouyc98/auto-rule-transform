from ruleparse import *
import ruleparse
from data import *
from ruleparse import RCTree
from owlready2 import *
import json
import jieba_fast as jieba
from gensim.models import Word2Vec
from scipy import spatial
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from collections import Counter
import math
from sklearn.decomposition import TruncatedSVD
import warnings
import ifc2ttl

warnings.filterwarnings("ignore", category=Warning)

CMP_DICT_Onto = OrderedDict([('<=', '小于等于 小于或等于 不大于 不高于 不多于 不超过 ≤'.split()),
                             ('>=', '大于等于 大于或等于 不小于 不低于 不少于 ≥'.split()),
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

TERM_CHANGE_DICT = OrderedDict(
    [('isFireProtectionSubdivision_Boolean', ['BuildingSpace', '建筑区域', 'isFireProtectionSubdivision_Boolean', '是防火分区']),
     ('IsSecurityExits_Boolean', ['Doors', '门', 'IsSecurityExits_Boolean', '是安全出口']),
     ('isFireWall_Boolean', ['Wall', '墙体', 'isFireWall_Boolean', '是防火墙'])])
EQUIVALENT_TERM_DICT = OrderedDict([('FireWall', ['Wall', '墙体', 'isFireWall_Boolean', '是防火墙']),
                                    ])
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
    0. keyword matching
    0.1. weight keyword matching
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

    if method == 0:
        for one_onto_description in onto_description:
            if word in one_onto_description:
                return 1
        else:
            return 0

    elif method == 0.1:
        similarity = 0
        words_num = len(onto_description)
        for one_onto_description in onto_description:
            if word == one_onto_description:
                similarity += 1
            elif word in one_onto_description:
                similarity += 0.6 / math.log(len(one_onto_description))
        return similarity / words_num

    elif method == 1:
        '''
        method = 1 This method calculate the similartiy then mean similarity
        '''
        input_word_seq = Sentence(word, stopwords).tokens_no_stop()
        caculation_times = 1
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
                        weights1.append(a / (a + dictionary.cfs[dictionary.token2id[token1]] / total_freq))
                    else:
                        weights1.append(a / (a + 1000 / total_freq))
                for token2 in tokens2:
                    if token2 in dictionary.token2id:
                        weights2.append(a / (a + dictionary.cfs[dictionary.token2id[token2]] / total_freq))
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
    onto_term: (term:str, similarity:float)
'''


def most_similar_onto_term(words: list, method=1):
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
    class_names = []
    dataproperty_names = []
    for word in words:
        similarity_scores = {}
        for one_class in ontology_class:
            class_name = one_class[0]
            class_names.append(class_name)
            class_description = one_class[2]
            similarity_scores[class_name] = word2vec_similarity(class_description, word, model, stopwords, dictionary,
                                                                method=method)
        for one_dataproperty in ontology_dataproperty:
            dataproperty_name = one_dataproperty[0]
            dataproperty_names.append(dataproperty_name)
            dataproperty_description = one_dataproperty[2]
            similarity_scores[dataproperty_name] = word2vec_similarity(dataproperty_description, word, model,
                                                                       stopwords, dictionary, method=method)
        similarity_scores_sort = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
        for key in similarity_scores_sort:
            onto_term = key
            type = ''
            if onto_term[0] in dataproperty_names:
                type = 'dataproperty'
            elif onto_term[0] in class_names:
                type = 'class'
            if type is not '':
                term_word_pair.append({"label": onto_term, "word": word, 'type': type})
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


'''
State: use
Function: This function set ontology class type and ontology class name for RCtree nodes based on similarity matching
link Method: 
    0. keyword matching
    0.1. weight keyword matching
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
'''


def RCNode_entity_link(link_method=0, islog=False):
    rcts = []

    def preorder(RCtree):
        words = [RCtree.curr_node.word]
        if words is not None:
            term_word_pair = most_similar_onto_term(words, method=link_method)
            RCtree.curr_node.set_onto_info(term_word_pair[0]["label"][0], term_word_pair[0]["type"])
        while RCtree.curr_node.has_child():
            for one_childnode in RCtree.curr_node.child_nodes:
                RCtree.curr_node = one_childnode
                preorder(RCtree)

    if islog:
        logger = Logger(file_name='rulegen.log', init_mode='w+')
        log = logger.log
    else:
        log = print
    n_parse = 0

    # rule classify model
    keyword_dict = init_classify_dict()

    for seq, label in seq_data_loader('text'):
        # rule classify
        rule_category = rule_classification(seq, keyword_dict, method=1)
        rct = RCTree(seq, label, log)
        rct.parse()
        preorder(rct)
        n_parse += 1
        rct.log_msg(n_parse)
        rct.set_rule_category(rule_category)
        rcts.append(rct)
    log('-' * 90)  # 这个log是必要的，因为在解析log文件时默认以90个'-'作为rctree信息之间的分隔符，最后一个rct log完后要补一个
    log('Rule gen complete.')
    return rcts


'''
State: use
Function: give a senquence, Determine the category of the code, add rule_category for rct
Categorys: 
    1. direct attribute constraint: 
        Length, width, height, thickness, depth, span, accuracy, fire resistance grade and other attributes that can be directly stored in the BIM model;
        Entities that can be directly obtained in the BIM model, such as a certain building, component, device, material, structural form, etc.
    2. indirect attribute constraint: 
        Quantity, distance, area, slope, existence, slenderness ratio, height-span ratio, bearing capacity, relative position and other attributes that need 
        to be calculated and analyzed in the BIM model (some describe the length, width, The provisions of higher attributes also require certain calculations);
    3. others
nethod:
    1. keyword matching
    2. LSTM (Todo)
'''


def init_classify_dict(file_path=r'..\data\rules\classify_keywords.txt'):
    with open(file_path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        direct_attr_dict = lines[1].rstrip().split(',')
        indirect_attr_dict = lines[3].rstrip().split(',')
        return [direct_attr_dict, indirect_attr_dict]


def init_model():
    pass


def rule_classification(sentences: str, model, method=1):
    if method == 1:
        assert type(model) == list, 'the model is not true!'
        direct_attr_dict = model[0]
        indirect_attr_dict = model[1]
        for keyword in indirect_attr_dict:
            if keyword in sentences:
                return 2
        for keyword in direct_attr_dict:
            if keyword in sentences:
                return 1
        return 3


'''
State: use
Function: log the augmentation rctree and the sparql
'''


# log tree after augmention
def log_rcts(rcts):
    logger = Logger(file_name='rulegen.log', init_mode='w+')
    log = logger.log
    n_parse = 0

    def log_rct(rct, n_parse):
        print('*' * 90)
        print('The rct after equivalent class define and dataproperty replacement is:')
        rct.change_log_func(log)
        rct.log_msg(n_parse)
        log('-' * 90)
        log('Rule gen complete.')

    for rct in rcts:
        log_rct(rct, n_parse)
        n_parse += 1


'''
State: use
Function: read txt to generate RCTree
'''


class logFile:
    SEP = '\n' + '-' * 90 + '\n'

    def __init__(self, file_txt):
        self.txt = file_txt
        self.msgs = self.txt.split(self.SEP)

    @staticmethod
    def index_keyword(lines, word: str):
        for index, line in enumerate(lines):
            if word in line:
                return index

    def get_rcts(self):
        rcts = []
        sen_classify_dict = init_classify_dict()
        for msg in self.msgs:
            rct = self.msg2Rctree(msg, sen_classify_dict)
            if rct is not None:
                rcts.append(rct)
        return rcts

    def msg2Rctree(self, msg: str, sen_classify_dict):
        """
            :param msg:
                    [0]#f57b9fd
                    Seq:	防火墙的耐火极限不低于3##h。
                    Label:	[防火墙/obj]的[耐火极限/prop][不低于/cmp][3##h/Rprop]。
                    RCTree:	#b466cd0
                    		[墙体:Wall]
                    		|-[耐火极限:hasFireResistanceLimits_hour] ≥ [3##h]
                    		|-?[是防火墙:isFireWall_Boolean] = [True]
                    Parsing complete
                    Sparql:
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                    PREFIX myclass: <http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#>
                    PREFIX : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#>
                    SELECT DISTINCT ?class_Wall_1 ?class_Wall_1_id
                    WHERE {
                    	?class_Wall_1 rdf:type owl:NamedIndividual , myclass:Wall .
                    	?class_Wall_1 :hasGlobalId ?class_Wall_1_id .
                    	?class_Wall_1 :hasFireResistanceLimits_hour ?dataproperty_hasFireResistanceLimits_hour_2 .
                    	BIND ((?dataproperty_hasFireResistanceLimits_hour_2 >= '3'^^xsd:decimal) AS ?Pass_hasFireResistanceLimits) .
                    	FILTER (?Pass_hasFireResistanceLimits = 'false'^^xsd:boolean) .
                    	?class_Wall_1 :isFireWall_Boolean ?dataproperty_isFireWall_Boolean_3 .
                    	BIND ((?dataproperty_isFireWall_Boolean_3 = 'true'^^xsd:boolean) AS ?Pass_isFireWall) .
                    	FILTER (?Pass_isFireWall = 'true'^^xsd:boolean) .
                    	}
            :return:
                one rct
        """
        if ']#' in msg and 'Seq' in msg:
            # === check & pre-process, move 'SyntaxError' to the second line
            lines = [l for l in msg.split('\n')]
            assert len(lines) > 5
            if lines[0].startswith('!SyntaxError'):
                msg = '\n'.join([lines[1]] + [lines[0]] + lines[2:])
            assert msg[0] == '[', f"msg format wrong! {msg}"

            lines = [l for l in msg.split('\n') if l.strip()]
            index_sparql = logFile.index_keyword(lines, 'Sparql')
            seq_id_, seq, label, rct, eval, sparql = lines[0], lines[1], lines[2], lines[3:index_sparql - 1], lines[
                index_sparql - 1], lines[index_sparql + 1:]

            # get idx and seq_id
            i = seq_id_.index('#')
            idx = int(seq_id_[1:i - 1])
            seq_id_ = seq_id_[i:]

            # get seq
            assert seq.startswith('Seq:\t')
            seq = seq[seq.index(':') + 2:]

            # get label
            assert label.startswith('Label:\t')
            label = label[label.index(':') + 2:]
            _, label_iit = slabel_to_seq_label_iit(label)

            assert rct[0].startswith('RCTree:\t')
            rct_id_ = rct[0][rct[0].index(':') + 2:]

            # get new RCTree
            new_rct = RCTree(seq, label_iit)

            pattern = re.compile(r'\[\S*\]')
            cmp_pattern = re.compile(r"\].*\[")

            for index, line in enumerate(rct[1:]):
                if index == 0:
                    sobj_strs = pattern.findall(line)
                    obj_str = sobj_strs[-1][1:-1]  # obj node
                    sobj_strs = sobj_strs[:-1]  # exclude obj node
                    obj_node = RCNode(obj_str.split(':')[0], 'obj')
                    obj_node.set_onto_info(obj_str.split(':')[2], obj_str.split(':')[1])
                    new_rct.obj_node = obj_node
                    new_rct.curr_node = new_rct.obj_node

                    if len(sobj_strs) > 0:
                        for sobj_str in sobj_strs[:-1]:
                            sobj_str = sobj_str[1:-1]
                            sobj_node = RCNode(sobj_str.split(':')[0], 'sobj')
                            sobj_node.set_onto_info(sobj_str.split(':')[2], sobj_str.split(':')[1])
                            new_rct.add_parent_node(sobj_node)
                    new_rct.add_parent_node(new_rct.root)
                    new_rct.curr_node = new_rct.obj_node

                else:
                    prop_strs = pattern.findall(line)
                    if '?' in line:
                        req_tag = 'ARprop'
                    else:
                        req_tag = 'Rprop'

                    if len(prop_strs) == 1:
                        prop_str = prop_strs[0][1:-1]
                        prop_node = RCNode(prop_str.split(':')[0], 'prop')
                        prop_node.set_onto_info(prop_str.split(':')[2], prop_str.split(':')[1])
                    else:
                        assert len(prop_strs) == 2, 'prop_str is wrong'
                        prop_str = prop_strs[0][1:-1]
                        req_str = prop_strs[1][1:-1]
                        prop_node = RCNode(prop_str.split(':')[0], 'prop')
                        prop_node.set_onto_info(prop_str.split(':')[2], prop_str.split(':')[1])
                        if req_str == 'True':
                            req_str = True
                        req_node = RCNode(req_str, req_tag)
                        cmp_word = cmp_pattern.findall(line)[0][2]
                        cmp_node = RCNode(cmp_word, 'cmp')
                        prop_node.set_req((cmp_node, req_node, None))
                    if '|--' not in line and '|-' in line:
                        new_rct.obj_node.add_child(prop_node)
                        new_rct.curr_node = prop_node
                    elif '|--' in line:
                        new_rct.curr_node.add_child(prop_node)
                        new_rct.curr_node = prop_node
            new_rct.seq_id = seq_id_
            new_rct.seq = seq
            new_rct.set_sparql(sparql)
            new_rct.set_rule_category(rule_classification(seq, sen_classify_dict, method=1))
            return new_rct
        return None


'''
State: use
Funtion: give a RCtree without entity link, generate a sparql rule.
'''


def sparql_generator(rct):
    # Determine if a dataproperty node needs a term replacement
    def ischange_dataproperty_node(node):
        if node.has_child():
            return True
        if node.onto_name in TERM_CHANGE_DICT:
            req = node.req[1].word
            if type(req) != type(True):
                return True
        return False

    def preorder_classdefine(RCtree, sparql_con):
        if RCtree.curr_node.onto_type is not None:
            # to add sparql pronoun for class node, for example ?element

            if RCtree.curr_node.onto_type == 'class':
                old_onto_classname = RCtree.curr_node.onto_name
                # replace by equivalent node
                if old_onto_classname in EQUIVALENT_TERM_DICT:
                    new_onto_classname, new_word, new_onto_dataproperty, new_reqword = EQUIVALENT_TERM_DICT[
                        old_onto_classname]
                    RCtree.curr_node.set_onto_info(new_onto_classname, 'class')
                    RCtree.curr_node.set_word(new_word)
                    # add a new node to store dataproperty
                    new_childnode = RCNode(new_reqword, 'prop')
                    new_childnode.set_onto_info(new_onto_dataproperty, 'dataproperty')
                    req_cmp_node = RCNode('=', 'cmp')
                    req_value_node = RCNode(True, 'ARprop')
                    new_childnode.set_req((req_cmp_node, req_value_node, None))
                    RCtree.curr_node.add_child(new_childnode)

                if not hasattr(RCtree.curr_node, 'sparql_pronoun'):
                    pronoun_count = RCtree.count_node_pronoun()
                    RCtree.curr_node.add_sparql_pronoun(pronoun_count)

                sparql_pronoun = RCtree.curr_node.sparql_pronoun
                sparql_con += sparql_pronoun + ' rdf:type owl:NamedIndividual , myclass:' + RCtree.curr_node.onto_name + ' .\n\t'
                sparql_con += sparql_pronoun + ' :hasGlobalId ' + sparql_pronoun + '_id .\n\t'

            # if a RCNode onto_type is a dataproperty and it has child, then its onto_name is wrong.
            # Its onto_type will be changed to class and a corresponding dataproperty child node will be add.
            elif RCtree.curr_node.onto_type == 'dataproperty' and ischange_dataproperty_node(RCtree.curr_node):
                old_onto_classname = RCtree.curr_node.onto_name

                # change wrong node
                if old_onto_classname in TERM_CHANGE_DICT:
                    new_onto_classname, new_word, new_onto_dataproperty, new_reqword = TERM_CHANGE_DICT[
                        old_onto_classname]
                    RCtree.curr_node.set_onto_info(new_onto_classname, 'class')
                    RCtree.curr_node.set_word(new_word)
                    # add a new node to store dataproperty
                    new_childnode = RCNode(new_reqword, 'prop')
                    new_childnode.set_onto_info(new_onto_dataproperty, 'dataproperty')
                    req_cmp_node = RCNode('=', 'cmp')
                    req_value_node = RCNode(True, 'ARprop')
                    new_childnode.set_req((req_cmp_node, req_value_node, None))
                    RCtree.curr_node.add_child(new_childnode)

                    if not hasattr(RCtree.curr_node, 'sparql_pronoun'):
                        pronoun_count = RCtree.count_node_pronoun()
                        RCtree.curr_node.add_sparql_pronoun(pronoun_count)

                    sparql_pronoun = RCtree.curr_node.sparql_pronoun
                    sparql_con += sparql_pronoun + ' rdf:type owl:NamedIndividual , myclass:' + RCtree.curr_node.onto_name + ' .\n\t'
                    sparql_con += sparql_pronoun + ' :hasGlobalId ' + sparql_pronoun + '_id .\n\t'

        while RCtree.curr_node.has_child():
            for one_childnode in RCtree.curr_node.child_nodes:
                RCtree.curr_node = one_childnode
                # need to return value otherwise it will return none
                return preorder_classdefine(RCtree, sparql_con)
        return sparql_con

    def preorder_relation(RCtree, sparql_con):
        if RCtree.curr_node.onto_type is not None:
            if RCtree.curr_node.onto_type == 'class':
                assert hasattr(RCtree.curr_node, 'sparql_pronoun'), 'The sparql class pronoun is not gen complete yet!'
                # only do it when the current node type = class
                for one_childnode in RCtree.curr_node.child_nodes:
                    if one_childnode.onto_type is not None:
                        if one_childnode.onto_type == 'dataproperty':
                            # to add sparl pronoun for dataproperty node, for example ?dataproperty
                            if not hasattr(one_childnode, 'sparql_pronoun'):
                                pronoun_count = RCtree.count_node_pronoun()
                                one_childnode.add_sparql_pronoun(pronoun_count)
                            req_cmp_rawdata = one_childnode.req[0].word
                            req_cmp = get_cmp_str_onto(req_cmp_rawdata)
                            req_value_rawdata = one_childnode.req[1].word
                            if req_value_rawdata == True:  # req_value_rawdata == True
                                req_value = req_value_rawdata
                                flag = "'true'"
                                sparql_con += RCtree.curr_node.sparql_pronoun + ' :' + one_childnode.onto_name + ' ' + one_childnode.sparql_pronoun + ' .\n\t'
                                sparql_con += 'BIND ((' + one_childnode.sparql_pronoun + ' ' + req_cmp + ' \'' + 'true' + '\'^^xsd:boolean) AS ?Pass_' + \
                                              one_childnode.sparql_pronoun.split('_')[
                                                  1] + ') .\n\t'  # 'BIND ((?dataproperty1 >= 'true'^^xsd:boolean) AS ?Pass1) .'
                                sparql_con += 'FILTER (?Pass_' + one_childnode.sparql_pronoun.split('_')[
                                    1] + " = " + flag + "^^xsd:boolean) .\n\t"  # 'FILTER (?Pass1 = 'true'^^xsd:boolean) .'

                            else:
                                req_value = re.findall(r"\d+\.?\d*", req_value_rawdata)[0]
                                flag = "'false'"
                                sparql_con += RCtree.curr_node.sparql_pronoun + ' :' + one_childnode.onto_name + ' ' + one_childnode.sparql_pronoun + ' .\n\t'
                                sparql_con += 'BIND ((' + one_childnode.sparql_pronoun + ' ' + req_cmp + ' \'' + req_value + '\'^^xsd:decimal) AS ?Pass_' + \
                                              one_childnode.sparql_pronoun.split('_')[
                                                  1] + ') .\n\t'  # 'BIND ((?dataproperty1 >= '2.0'^^xsd:decimal) AS ?Pass1) .'
                                sparql_con += 'FILTER (?Pass_' + one_childnode.sparql_pronoun.split('_')[
                                    1] + " = " + flag + "^^xsd:boolean) .\n\t"  # 'FILTER (?Pass1 = 'false'^^xsd:boolean) .'


                        elif one_childnode.onto_type == 'class':
                            domain_class = RCtree.curr_node.onto_name
                            range_class = one_childnode.onto_name
                            object_property = ifc2ttl.Building_element.get_objprop(domain_class, range_class)
                            sparql_con += RCtree.curr_node.sparql_pronoun + ' :' + object_property + ' ' + one_childnode.sparql_pronoun + ' .\n\t'

        while RCtree.curr_node.has_child():
            for one_childnode in RCtree.curr_node.child_nodes:
                RCtree.curr_node = one_childnode
                return preorder_relation(RCtree, sparql_con)
        return sparql_con

    # return the node that meet requirements
    def specify_node(rct, tag, onto_type):
        if rct.curr_node.tag == tag and rct.curr_node.onto_type == onto_type:
            return rct.curr_node
        while rct.curr_node.has_child():
            for one_childnode in rct.curr_node.child_nodes:
                rct.curr_node = one_childnode
                return specify_node(rct, tag, onto_type)

    '''
    State: use
    function: add prefix and suffix, select the most important element 
    rct.rule_category:
        1: direct constraint
        2: indirect constraint, now only contains quantifier constraints
    '''

    def prefix_suffix(rct, sparql_con):
        sparql_prefix = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nPREFIX myclass: <http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#>\nPREFIX : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#>\n'
        sparql_suffix = '}\n'
        if rct.rule_category == 1:
            key_node = rct.obj_node
            sparql_prefix += 'SELECT DISTINCT ' + key_node.sparql_pronoun + ' ' + key_node.sparql_pronoun + '_id\nWHERE {\n\t'
        elif rct.rule_category == 2:
            key_node = rct.obj_node
            rct.curr_node = rct.root
            number_node = specify_node(rct, 'prop', 'class')
            if number_node is not None:
                req_cmp_rawdata = number_node.req[0].word
                req_cmp = get_cmp_str_onto(req_cmp_rawdata)
                req_value_rawdata = number_node.req[1].word
                req_value = re.findall(r"\d+\.?\d*", req_value_rawdata)[0]

                sparql_prefix += 'SELECT ' + key_node.sparql_pronoun + ' ' + key_node.sparql_pronoun + '_id ' + '(COUNT(distinct ' + number_node.sparql_pronoun + ') AS ' + number_node.sparql_pronoun + '_num)\n'
                sparql_prefix += 'WHERE {\n\t'
                sparql_suffix += 'GROUP BY ' + key_node.sparql_pronoun + ' ' + key_node.sparql_pronoun + '_id\n'
                sparql_suffix += 'HAVING (' + number_node.sparql_pronoun + '_num ' + req_cmp + ' ' + req_value + ')\n'
                sparql_suffix += 'ORDER BY DESC (' + number_node.sparql_pronoun + '_num)'
            else:
                sparql_prefix += 'SELECT DISTINCT *\nWHERE {\n\t'
        else:
            sparql_prefix += 'SELECT DISTINCT *\nWHERE {\n\t'
        return sparql_prefix + sparql_con + sparql_suffix

    rct.curr_node = rct.root
    sparql_con = ''
    sparql_con = preorder_classdefine(rct, sparql_con=sparql_con)
    rct.curr_node = rct.root
    sparql_con = preorder_relation(rct, sparql_con=sparql_con)
    # sparql_prefix = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nPREFIX myclass: <http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#>\nPREFIX : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#>\n' \
    #                 'SELECT DISTINCT *\nWHERE {\n\t'
    # sparql_full = sparql_prefix + sparql_con + '}\n'
    sparql_full = prefix_suffix(rct, sparql_con)
    rct.set_sparql(sparql_full)
    # print rct after the augmented
    return sparql_full


'''
State: discard
Function: This function set ontology class type and ontology class name for RCtree nodes based on Keywords_dict
Now it is replaced by the function RCNode_entity_link
'''


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


'''
State: discard
Function: This class is replaced by sparql_generator
'''


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
    # docanno_src = '..\data\docanno\FireCode_label_merge.json'
    # __test_for_el(docanno_src=docanno_src, method=2)

    '''
        pre-train for tf-idf
    '''
    # rules_TFIDF()
    # dictionary = corpora.Dictionary.load(r"./models/tfidf/rules_doc2bow.dict")
    # model_load = models.TfidfModel.load(r"./models/tfidf/rules_tfidf.word2vec_model")
    # stoplist = stopwordslist(r'.\models\word2vec\Stopwords.txt')
    # sentence1 = Sentence('生产的火灾危险性类别为乙级，厂房的耐火等级为二级的高层厂房，面积不超过1500m2。', stoplist)
    # print(sentence1.tfidf_weight(dictionary, model_load))

    '''
        test for sparql generator
    '''
    # rcts = RCNode_entity_link(2, False)
    # for rct in rcts:
    #     sparql_generator(rct)
    # log_rcts(rcts)

    '''
       test for logFile
    '''
    with open(r'./logs/rulegen.log', 'r') as f1:
        rulegen_txt = f1.read()
    newlog = logFile(rulegen_txt)
    rcts = newlog.get_rcts()
    for rct in rcts:
        print(sparql_generator(rct))

