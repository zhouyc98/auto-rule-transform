# Auto Rule Transform

Auto Rule Transform: Automated rule transformation for automated rule checking.  


## Data Information

In data/xiaofang/[sentences_all.json](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/data/xiaofang/sentences_all.json), it contains all sentences with labels developed in this research.  

In src/logs/[rulecheck-eval-v50.log](https://github.com/Zhou-Yucheng/auto-rule-transform/blob/main/src/logs/rulecheck-eval-v50.log), it shows the parsing result of these sentences in a text-based format.  

## Setup & Usage

- Clone or download the repo  
  ```
  git clone https://github.com/Zhou-Yucheng/auto-rule-transform.git
  cd auto-rule-transform
  ```
- Install the requirements  
  ```
  pip install -r requirements.txt
  ```

### Semantic labeling

This repo uses [Pytorch](https://pytorch.org/) for training deep learning models. You can follow the official [get-started](https://pytorch.org/get-started/locally/) to install it.

Note: if you want to train the model using 16bit-float acceleration, please ensure your Pytorch version >= 1.6, because we use the Pytorch native module [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) which is introduced in Pytorch 1.6. Otherwise, you may would like to comment the `from torch.cuda.amp import autocast, GradScaler` line and remove two `with autocast()` statements in train.py.

Run train.py, and then you will get the trained model in src/models and the log file in src/logs/train.log.

  ```
cd src
python3 train.py
  ```
For more information about usages, run `python3 train.py -h`  

The best model trained in this research is BertZh-f777029c.pth, which can be downloaded from [Google Drive](https://drive.google.com/file/d/1hwm9h0Z-ocNijgLmbBltmarFe3CAmAbt/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1iq1_13DHfZZrH6Z5TBrg0Q) (提取码 8hys).  
If you want to report the performance of it, put it in src/models/ and run:

  ```
cp models/BertZh-f777029c.pth models/_BertZh0_best.pth
python3 train.py --report 
# then it will automatically report the model named _BertZh0_best.pth
  ```

### Parsing

Run rulecheck.py, and then you will get a new rulecheck.log & rulecheck-eval-v51.log in src/logs/  
  ```
cd src
python3 rulecheck.py
  ```

If you want to perform interactively rule transformation, run:
  ```
python3 rulecheck.py -i
# then, input the id of the sentence (see data/xiaofang/sentence_all.json), it will read the sentence and show the parsing result immediately
  ```
