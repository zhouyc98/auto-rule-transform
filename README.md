# Auto Rule Transform

Automated rule transformation for automated rule checking.  

This repo contains the dataset, codes, and documents for the following paper:  

* Y.-C. Zhou, Z. Zheng, J.-R. Lin, X.-Z. Lu, Integrating NLP and context-free grammar for complex rule interpretation towards automated compliance checking, Computers in Industry. 142 (2022) 103746. https://doi.org/10.1016/j.compind.2022.103746.
* Z. Zheng, Y.-C. Zhou, X.-Z. Lu, J.-R. Lin, Knowledge-informed semantic alignment and rule interpretation for automated compliance checking, Automation in Construction. 142 (2022) 104524. https://doi.org/10.1016/j.autcon.2022.104524.

![example](src/logs/example.jpg)




## Dataset Information
The data/xiaofang/[sentences_all.json](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/data/xiaofang/sentences_all.json) contains all sentences (in Chinese) with labels developed in this research.  

The src/logs/[ruleparse-eval.log](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/src/logs/ruleparse-eval.log) stores the parsing result (in Chinese) of the dataset in a text-based format (note: VSCode user can install the Log File Highlighter extension and configure it with [log-file-highlighter.txt](src/logs/log-file-highlighter.txt) to enable our customized syntax highlight).  

The data/docanno/[FireCode_label_merge.json] contains the semantic alignment labels developed in the research.

The data/rules/[建筑设计防火规范-第三章语料-class.txt] contains the text classification labels developed in the research.




## Semantic Labeling

This repo uses [Pytorch](https://pytorch.org/) for training deep learning models. Please ensure Pytorch's version >= 1.6 because we use [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) introduced in v1.6 for FP16 acceleration in CUDA device. To use FP32 in CPU device, specify arg `--fp16 0` in train.py.

This repo uses BERT model provided by [transformers (v2.11)](https://pypi.org/project/transformers/2.11.0/)  package. By default, the model training uses [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main), please download pytorch_model.bin and other json & txt files [here](https://huggingface.co/bert-base-chinese/tree/main) and place them in src/models/bert-base-chinese/.

Run train.py in src/ for model training, which will store trained models in src/models/:

  ```
python3 train.py
  ```
For more information about usages, run `python3 train.py -h`  

To report performance of the model _BertZh0_best.pth, run `python3 train.py --report`

Run inference.py for semantic labeling , which will use model _BertZh0_best.pth and read all txt files in data/xiaofang/test and store the labeling result in src/logs/predictions:

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



## Citation

```
@article{zhou2022a,
	author = {Yu-Cheng Zhou and Zhe Zheng and Jia-Rui Lin and Xin-Zheng Lu},
	title = {Integrating {NLP} and context-free grammar for complex rule interpretation towards automated compliance checking},
	journal = {Computers in Industry},
	volume = {142},
	pages = {103746},
	year = {2022},
	doi = {https://doi.org/10.1016/j.compind.2022.103746},
}

@article{zheng2022a,
	author = {Zhe Zheng and Yu-Cheng Zhou and Xin-Zheng Lu and Jia-Rui Lin},
	title = {Knowledge-informed semantic alignment and rule interpretation for automated compliance checking},
	journal = {Automation in Construction},
	volume = {142},
	pages = {104524},
	year = {2022},
	doi = {https://doi.org/10.1016/j.autcon.2022.104524},
}
```



## License

This project is free and open source for universities, research institutes, enterprises and individuals for research purposes only, and the commercial purpose is not permitted.  
本项目面向大学、研究所、企业以及个人用于研究目的免费开放源代码，不得将其用于任何商业目的。
