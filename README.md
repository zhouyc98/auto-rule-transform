# Auto Rule Transform

Automated rule transformation for automated rule checking.  




## Dataset Information

In data/xiaofang/[sentences_all.json](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/data/xiaofang/sentences_all.json), it contains all sentences with labels developed in this research.  

In src/logs/[rulecheck-eval.log](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/src/logs/rulecheck-eval.log), it shows the parsing result of these sentences in a text-based format.  

In data/docanno/[FireCode_label_merge.json], it contains the semantic alignment labels developed in the research

In data/rules/[建筑设计防火规范-第三章语料-class.txt], it contains the text classification labels developed in the research
## Setup

- Clone or download the repo  
  ```
  git clone https://github.com/Zhou-Yucheng/auto-rule-transform.git
  cd auto-rule-transform
  ```
- Install the requirements 
  ```
  pip install -r requirements.txt
  ```

## Semantic labeling

This repo uses [Pytorch](https://pytorch.org/) for training deep learning models. You can follow the official [get-started](https://pytorch.org/get-started/locally/) to install it.

Note: if you want to use FP16 acceleration to train the model, please ensure your Pytorch version >= 1.6, because we use the Pytorch native module [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) which is introduced in Pytorch 1.6. Otherwise, you may would like to comment the `from torch.cuda.amp import autocast, GradScaler` and remove relevant statements in train.py.

Run train.py in src/, and then you will get the trained model in src/models/ and the log file in src/logs/train.log.

  ```
python3 train.py
  ```
For more information about usages, run `python3 train.py -h`  

To report the performance of a model in src/models/, rename it to _BertZh0_best.pth and run:

  ```
python3 train.py --report # it will report the model named _BertZh0_best.pth
  ```

## Parsing

Run ruleparse.py in src/, and then it will read the sentences in data/xiaofang/sentences_all.json and log the result in src/logs/ruleparse.log & src/logs/ruleparse-eval.log

  ```
python3 ruleparse.py
  ```

If you want to change the dataset of parsing to data/xiaofang/sentences.txt, use the -d argument to specify:

  ```
python3 ruleparse.py -d text
  ```

To generate the [XML check set](https://interoperability.autodesk.com/modelcheckerconfigurator/downloads/xmlschema.pdf) rules for [Autodesk Revit model checker](https://interoperability.autodesk.com/modelchecker.php) after the parsing, add -g switch (in beta version now):

  ```
python3 ruleparse.py -d text -g
  ```

If you want to perform interactive (semi-automated) rule transformation, run:

  ```
python3 ruleparse.py -i
# then, input the id of a sentence (see data/xiaofang/sentence_all.json),  
# it will read the sentence and show the parsing result immediately
  ```

## SPARQL generation

This function is used to generate SPARQL codes, which can be reasoned by protege, from the Semantic labeling results (i.e., data/xiaofang/sentences.txt)

The unsupervised learning-based semantic alignment methods (e.g., the word2vec techniques) and rule-based conflict resolution methods are used.

The following steps are required to generate SPARQL automatically.

(1) First you should download the word2vec model from `https://pan.baidu.com/s/1MEz7UJqhP0RdEMNqZCBpaQ password: 49tp`, and release them in src/models/

(2) Then you should put the input text file into data/xiaofang/sentences.txt

(3) Assert the BuildingDesignFireCodesOntology.pkl and BuildingDesignFireCodesOntology.owl are in data/ontology/, which is used for semantic alignment

(4) Assert the classify_keywords.txt is in data/rules, which is used for text classification

(5) Run `python rulegen.py` to generate SPARQL. The generated file is in src/logs/rulegen.log


## References

If you used the function. please cite these articles: 

    {
         author = {Yucheng Zhou, Zhe Zheng, Jiarui Lin and Xinzheng Lu},
         title = {Integrating NLP and Context-Free Grammar for Complex Rule Interpretation towards Automated Compliance Checking},   
         href = https://doi.org/10.13140/RG.2.2.22993.45921 
         year = {2021}
     }
    {
         author = {Zhe Zheng, Yucheng Zhou, Xinzheng Lu and Jiarui Lin},
         title = {Knowledge-Informed Semantic Alignment and Rule Interpretation for Automated Compliance Checking},
         year = {2021}
     }
