# Auto Rule Transform

Automated rule transformation for automated rule checking.  

This repo contains the dataset, codes, and documents for the paper entitled "Integrating NLP and Context-Free Grammar for Complex Rule Interpretation towards Automated Compliance Checking" (DOI: http://dx.doi.org/10.13140/RG.2.2.22993.45921).  

![example](src/logs/example.jpg)




## Dataset Information
The data/xiaofang/[sentences_all.json](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/data/xiaofang/sentences_all.json) contains all sentences (in Chinese) with labels developed in this research.  

The src/logs/[ruleparse-eval.log](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/src/logs/ruleparse-eval.log) stores the parsing result (in Chinese) of the dataset in a text-based format (note: VSCode user can install the Log File Highlighter extension and configure it with [log-file-highlighter.txt](src/logs/log-file-highlighter.txt) to enable our customized syntax highlight).  

The data/docanno/[FireCode_label_merge.json] contains the semantic alignment labels developed in the research.

The data/rules/[建筑设计防火规范-第三章语料-class.txt] contains the text classification labels developed in the research.



## Semantic Labeling

This repo uses [Pytorch](https://pytorch.org/) for training deep learning models. You can follow the official [get-started](https://pytorch.org/get-started/locally/) to install it.

Note: if you want to use FP16 acceleration to train the model, please ensure your Pytorch version >= 1.6 because we use [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) introduced in Pytorch 1.6. Otherwise, you may would like to comment the `from torch.cuda.amp import autocast, GradScaler` and remove relevant statements in train.py.

Run train.py in src/ for model training, which will store trained models in src/models/:

  ```
python3 train.py
  ```
For more information about usages, run `python3 train.py -h`  

To report performance of the model _BertZh0_best.pth, run `python3 train.py --report`

Run inference.py for semantic labeling , which will read all txt files in data/xiaofang/test and store the labeling result in src/logs/predictions:

  ```
python3 inference.py
  ```

## Syntactic Parsing

Run ruleparse.py in src/ for syntactic parsing, which will read sentences in data/xiaofang/sentences_all.json and store the result in src/logs/ruleparse-eval.log:

  ```
python3 ruleparse.py -d json
  ```

To change the dataset of parsing to data/xiaofang/sentences.txt, use the -d argument to specify:

  ```
python3 ruleparse.py -d text
  ```

To generate the [XML check set](https://interoperability.autodesk.com/modelcheckerconfigurator/downloads/xmlschema.pdf) rules for [Autodesk Revit model checker](https://interoperability.autodesk.com/modelchecker.php) after the parsing, add -g switch (in beta version now):

  ```
python3 ruleparse.py -d text -g
  ```

To perform interactive rule transformation, run:

  ```
python3 ruleparse.py -i
# then input the id of a sentence (ref data/xiaofang/sentence_all.json),  
# it will read the sentence and show the parsing result immediately
  ```

## Rule Generation (SPARQL)

This function is used to generate SPARQL codes, which can be reasoned by protege, from the semantic labeling results (i.e., data/xiaofang/sentences.txt)

The unsupervised learning-based semantic alignment methods (e.g., the word2vec techniques) and rule-based conflict resolution methods are used.

The following steps are required to generate SPARQL automatically.

1. Download the word2vec model from https://pan.baidu.com/s/1MEz7UJqhP0RdEMNqZCBpaQ (password: 49tp), and release them in src/models/

2. Put the input text file into data/xiaofang/sentences.txt

3. Make sure BuildingDesignFireCodesOntology.pkl and BuildingDesignFireCodesOntology.owl are in data/ontology/, which is used for semantic alignment; and classify_keywords.txt is in data/rules, which is used for text classification

4. Run `python rulegen.py` to generate SPARQL. The generated file is in src/logs/rulegen.log


## References

If you use this repo, please cite these articles: 
```
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
```

## License

This project is free and open source for universities, research institutes, enterprises and individuals for research purposes only, and the commercial purpose is not permitted.  
本项目面向大学、研究所、企业以及个人用于研究目的免费开放源代码，不得将其用于任何商业目的。
