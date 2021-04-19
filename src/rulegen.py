from ruleparse import *
import ruleparse
from owlready2 import *
import json
import jieba_fast as jieba
from gensim.models import Word2Vec

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
Function: give a ontology word and the corresponding set of description, calculate the similarity between a natural language word and the ontology word
Inout: 
    onto_description: list(str)
    word: str
    model: pretrained word2vec model
    stopword: stopwords list
output:
    similarity: float
'''


def similarity_onto_term(onto_description, word, model, stopwords):
    if len(onto_description) == 0:
        return 0
    word_seq = jieba.cut(word)
    caculation_times = 0
    similarity = 0
    for one_onto_description in onto_description:
        description_seq = jieba.cut(one_onto_description)
        for word_seq_item in word_seq:
            for description_seq_item in description_seq:
                # Eliminate the influence of meaningless words
                if not is_stop_words(description_seq_item, stopwords):
                    if word_seq_item in model and description_seq_item in model:
                        similarity += model.wv.similarity(word_seq_item, description_seq_item)
                        caculation_times += 1
    return similarity / caculation_times


'''
State: use
Function: give a word list, for each word find the most similar term in ontology
Input:
    word: str
    ontology: pickle file
output:
    onto_term: str
'''


def most_similar_onto_term(words):
    jieba.load_userdict(r'.\models\word2vec\wordsList500.txt')
    stopwords = stopwordslist(r'.\models\word2vec\Stopwords.txt')
    model = Word2Vec.load(r'.\models\word2vec\Merge.model')
    onto_file = r'..\data\ontology\BuildingDesignFireCodesOntology.pkl'

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
            similarity_scores[class_name] = similarity_onto_term(class_description, word, model, stopwords)
        for one_dataproperty in ontology_dataproperty:
            dataproperty_name = one_dataproperty[0]
            dataproperty_description = one_dataproperty[2]
            similarity_scores[dataproperty_name] = similarity_onto_term(dataproperty_description, word, model,
                                                                        stopwords)
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


def __test_for_el(docanno_src='./data/FireCode_label_merge.json'):
    with open(docanno_src, 'r', encoding='utf-8') as f1:
        test_data = json.load(f1)
    annotations = []
    for sentence in test_data:
        for annotation in sentence['annotations']:
            annotations.append(annotation)
    words = []
    for annotation in annotations:
        words.append(annotation['word'])
    predicitons = most_similar_onto_term(words)
    assert len(predicitons) == len(annotations), 'Somewords in annotations test dataset is not predict'
    true_num = 0
    for index in range(len(annotations)):
        assert predicitons[index]['word'] == annotations[index][
            'word'], 'the prediction word is not the same as that in test dataset'
        # data structure:
        #   predictions {'label':(term,similarity_score), 'word':word}
        #   annotaions {'label':term, 'word':word}
        if predicitons[index]['label'][0] == annotations[index]['label']:
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
    __test_for_el(docanno_src=docanno_src)
