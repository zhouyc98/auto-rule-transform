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

To report the performance of a model in src/models/, rename it to _BertZh0_best.pth and run:

  ```
python3 train.py --report # it will report the model named _BertZh0_best.pth
  ```

### Parsing

Run ruleparse.py, and then you will get a new ruleparse.log & ruleparse-eval.log in src/logs/  
  ```
cd src
python3 ruleparse.py
  ```

If you want to perform interactively rule transformation, run:
  ```
python3 ruleparse.py -i
# then, input the id of the sentence (see data/xiaofang/sentence_all.json), it will read the sentence and show the parsing result immediately
  ```
