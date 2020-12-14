# coding: utf-8

import argparse
import shutil
from torch import true_divide
import torch.nn as nn
import random
import csv
import pathlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from data import *
from utils import *
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # print less verbose log
SEED = 2020
if SEED >= 0:
    # seed_everything
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _show_param_distribution():
    gradss = []
    for p in filter(lambda p: p.grad is not None, model.parameters()):
        # p.shape: [x, y], [x]
        gradss.append(p.grad.detach().cpu().numpy().reshape(-1))
    grads = np.concatenate(gradss)

    grads = np.log(np.abs(grads) + 1e-10)

    plt.figure()
    plt.hist(grads, bins=10)
    plt.show()

    print(' Grad < clip_grad % = {}, LOG Grad max = {:.4f}, Grad norm2 = {:.4f}'
          .format(np.sum(grads < grad_norm_clip) / grads.size, np.max(grads), np.linalg.norm(grads)))


def _join_ljust(words, width=9):
    """join list of str to fixed width, left just"""
    return ' '.join(map(lambda s: s.ljust(width), words)).strip()


def plot_model_history(log_path, plot_loss=True):
    train_loss, train_f1, val_loss, val_f1 = [], [], [], []
    with open(log_path) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if (not row) or row[0] == 'Epoch':
                continue
            if '#' in row[0]:
                break
            row = list(map(float, row))
            train_loss.append(row[1])
            train_f1.append(row[2])
            val_loss.append(row[3])
            val_f1.append(row[4])

    plt.plot(train_f1, linewidth=0.5)
    plt.plot(val_f1)
    plt.title('Model F1w')
    plt.ylabel('F1w')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    if plot_loss:
        plt.figure()
        plt.plot(train_loss, linewidth=0.5)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()


def clear_prediction_logs(is_test=False):
    def _fn_flag(fn_dir_, fn_):
        if not os.path.isfile(os.path.join(fn_dir_, fn_)):
            return False

        if is_test:
            return '-test' in fn_.lower()
        else:
            return '-train' in fn_.lower() or '-val' in fn_.lower()

    pathlib.Path('./logs/predictions').mkdir(exist_ok=True)
    pathlib.Path('./logs/predictions/bak').mkdir(exist_ok=True)

    log_dir = './logs/predictions/'
    log_bak_dir = log_dir + 'bak/'

    for fn in os.listdir(log_bak_dir):
        if _fn_flag(log_bak_dir, fn):
            os.remove(log_bak_dir + fn)

    for fn in os.listdir(log_dir):
        if _fn_flag(log_dir, fn):
            shutil.move(log_dir + fn, log_bak_dir + fn)


def log_predictions(inputs, labels, preds, comment='train', is_test=False, corpus_=None):
    """inputs: shape: [bs, sql]; labels: shape: [bs*sql]"""
    bs = inputs.shape[0]
    preds = preds.view(bs, -1)
    if labels is not None:
        labels = labels.view(bs, -1)

    # ===== import-friendly
    is_print = is_test or args.verbose >= 3
    is_log = is_test or args.verbose >= 1
    if corpus_ is None:
        corpus_ = corpus
    # =====

    log_lines = []
    p_rand = random.random()
    n_log = bs  # logged number of seq
    for i in range(n_log):
        seq = inputs[i]
        label = labels[i] if labels is not None else None
        pred = preds[i]
        label_str, pred_str, seq_str = corpus_.render_seq_labels(seq, label, pred)

        if is_print and p_rand < 0.05 and i < 3:
            if is_test:
                print(f'{pred_str}\n')
            else:
                print(f'label: {label_str}')
                print(f'pred : {pred_str}\n')

        if is_log:
            log_lines.append(f'seq  : {seq_str}  \n')

            if is_test:
                log_lines.append(f'{pred_str}  \n\n')  # markdown needs 2 space before \n
            else:
                log_lines.append(f'label: {label_str}  \n')
                log_lines.append(f'pred : {pred_str}  \n\n')

    if is_log:
        # date_now = datetime.now().strftime('%m%d')
        with open(f"./logs/predictions/predictions-{comment}.log", 'a+', encoding='utf8') as f_:
            f_.write(f"-----[{datetime.now().isoformat()}]  \n\n")
            f_.writelines(log_lines)
            f_.write('  \n')


def cross_entropy_loss_non_pad(outputs, labels):
    """
    calculate the cross entropy loss, ignoring PAD token/tag

    ref: https://cs230.stanford.edu/blog/namedentity/#writing-a-custom-loss-function
        https://blog.csdn.net/qq_22210253/article/details/85229988

    :param outputs: [bs, sql, n_tags]
    :param labels:  [bs, sql]
    :return: single value
    """

    outputs = outputs.view(-1, outputs.shape[-1])  # [bs*sql, n_tags]
    outputs = F.log_softmax(outputs, dim=1)

    labels = labels.view(-1)  # [bs*sql]
    mask = (labels >= 0).float()  # bool -> float (0. or 1.)

    # pick the values corresponding to labels and multiply by mask
    output_scores = outputs[range(outputs.shape[0]), labels] * mask

    n_tokens = int(torch.sum(mask).item())  # num of non-pad tokens
    return -torch.sum(output_scores) / n_tokens


def cross_entropy_loss_non_pad_focal(outputs, labels, gamma=2):
    """
    calculate the cross entropy loss, ignoring PAD token/tag

    ref: https://cs230.stanford.edu/blog/namedentity/#writing-a-custom-loss-function
        https://blog.csdn.net/qq_22210253/article/details/85229988

    :param outputs: [bs, sql, n_tags]
    :param labels:  [bs, sql]
    :return: single value
    """

    # gamma=args.loss_gamma

    outputs = outputs.view(-1, outputs.shape[-1])  # [bs*sql, n_tags]
    outputs = F.softmax(outputs, dim=1)
    pts = outputs[range(outputs.shape[0]), labels]
    focal_loss = ((1 - pts) ** gamma) * pts.log()

    labels = labels.view(-1)  # [bs*sql]
    mask = (labels >= 0).float()  # bool -> float (0. or 1.)
    output_scores = focal_loss * mask

    n_tokens = int(torch.sum(mask).item())  # num of non-pad tokens
    return -torch.sum(output_scores) / n_tokens


def evaluate(log_preds=''):
    model.eval()
    with torch.no_grad():
        history = NNFullHistory()
        data_loader = tqdm(val_data_loader) if show_progressbar else val_data_loader
        for inputs, att_mask, labels in data_loader:
            # inputs: token_ids, labels: tag_ids
            inputs, att_mask, labels = inputs.to(device), att_mask.to(device), labels.to(device).view(-1)

            outputs = model(inputs, att_mask)  # shape: [bs, sql, n_tags]
            outputs = outputs.view(-1, outputs.shape[-1])  # [bs*sql, n_tags]
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)  # return (value, index)

            history.append(loss, predictions, labels, ignore_label=-1)
            if log_preds:
                log_predictions(inputs, labels, predictions, log_preds)
            if show_progressbar:
                p, r, f1 = history.avg_prf1_weight()
                data_loader.set_postfix({'loss': history.avg_loss(), 'f1': f1})

    return history


def train():
    """ train model in an epoch """
    model.train()
    history = NNFullHistory()
    data_loader = tqdm(train_data_loader) if show_progressbar else train_data_loader
    for inputs, att_mask, labels in data_loader:
        # inputs: token_ids, shape: [bs, sql]; labels: tag_ids, shape: [bs*sql]
        inputs, att_mask, labels = inputs.to(device), att_mask.to(device), labels.to(device).view(-1)

        optimizer.zero_grad()
        outputs = model(inputs, att_mask)  # shape: [bs, sql, n_tags]
        outputs = outputs.view(-1, outputs.shape[-1])  # [bs*sql, n_tags]
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        history.append(loss, predictions, labels, ignore_label=-1)
        # log_predictions(inputs, labels, predictions, 'train')
        if show_progressbar:
            p, r, f1 = history.avg_prf1_weight()
            data_loader.set_postfix({'loss': history.avg_loss(), 'f1': f1})

    return history


def log_and_save():
    global best_val_loss, best_val_history, best_epoch

    train_loss, val_loss = train_history.avg_loss(), val_history.avg_loss()
    train_p, train_r, train_f1 = train_history.avg_prf1_weight()
    val_p, val_r, val_f1 = val_history.avg_prf1_weight()
    best_prefix = '*' if (not best_val_history or val_f1 > best_val_history.avg_prf1_weight()[-1]) else ''

    # ===Print& Log
    log("train: \tloss={:.3f}, f1w={:.3f}".format(train_loss, train_f1))
    log(best_prefix + "val: \t  loss={:.3f}, f1w={:.3f}".format(val_loss, val_f1))

    # ===Save
    torch.save(model.state_dict(), f'./models/_BertZh{args.cuda}.pth')
    if best_prefix:
        best_val_loss = val_loss
        best_val_history = val_history
        best_epoch = epoch
        os.replace(f'./models/_BertZh{args.cuda}.pth', f'./models/_BertZh{args.cuda}_best.pth')
        torch.save(model.state_dict(), f'./models/{model_fullname} -f1w {val_f1:.3f} -ep {epoch:02d}.pth')
        for path in sorted(glob(f'./models/{model_fullname} -f1w*.pth'))[:-1]:
            os.remove(path)

    # ===Writer
    if epoch == 1:
        fwriter.write('Epoch, Train Loss, Train F1w, Valid Loss, Valid F1w, lr\n')
    fwriter.write('{}, {}, {}, {}, {}, {}\n'.format(epoch, train_loss, train_f1, val_loss, val_f1, lr))

    swriter.add_scalar('Train/Loss', train_loss, epoch)
    swriter.add_scalar('Train/F1w', train_f1, epoch)
    swriter.add_scalar('Train/Precision', train_p, epoch)
    swriter.add_scalar('Train/Recall', train_r, epoch)
    swriter.add_scalar('Valid/Loss', val_loss, epoch)
    swriter.add_scalar('Valid/F1w', val_f1, epoch)
    swriter.add_scalar('Valid/Precision', val_p, epoch)
    swriter.add_scalar('Valid/Recall', val_r, epoch)

    # ===Print result, Close writer
    if epoch == n_epoch:  # the last epoch
        best_val_p, best_val_r, best_val_f1 = best_val_history.avg_prf1_weight()
        log(f'\n*Best Valid: loss={best_val_loss:.3f}, f1w={best_val_f1:.3f}. At epoch: {best_epoch}')
        fwriter.write('\n#Time cost, ' + get_elapsed_time(start_time))
        fwriter.close()
        swriter.close()


def report_model(history_path=None, val_history=None, log_all_preds=False):
    """if given val_history, all other params are ignored"""
    log('\n=== Report ===')
    if val_history is None:
        global epoch, show_progressbar, model_fullname
        epoch = -1  # means newly loaded model
        load_last_best(model)
        val_history = evaluate()
        model_fullname = 'BertZh*'

    val_loss = val_history.avg_loss()
    val_p, val_r, val_f1 = val_history.avg_prf1_weight()
    report_table = val_history.avg_prf1_all(output_dict=False, label_tags=corpus.tags)

    log(f'[{model_fullname} -f1w {val_f1:.3f} -ep {epoch:02d}]')
    log(f"\nValid report:\n{report_table}")
    # log(f"\nValid report (binary-classification): loss={val_loss:.3f},")

    if log_all_preds:
        # clear log
        try:
            os.remove('./logs/predictions/predictions-train.log')
            os.remove('./logs/predictions/predictions-val.log')
        except OSError:
            pass

        # monkey patching
        print('\nLoging all predictions...')
        global val_data_loader, train_data_loader
        evaluate(log_preds='val')
        _val_data_loader = val_data_loader
        val_data_loader = train_data_loader
        evaluate(log_preds='train')
        val_data_loader = _val_data_loader

    if history_path is not None:
        plot_model_history(history_path)


def load_last_best(model_):
    path = f'./models/_BertZh{args.cuda}_best.pth'
    model_.load_state_dict(torch.load(path))
    print(f'Load model successfully in {path}')


def log(msg, end='\n'):
    # import-friendly (True for import)
    is_print = args.verbose >= 1 if 'args' in globals() else True

    if is_print:
        print(msg, end=end)
    with open(f'./logs/train.log', 'a+') as f_:
        f_.write(msg)
        f_.write(end)


def get_args():
    parser = argparse.ArgumentParser(description='NLP NER Project')

    # Default value
    _bs = 2 if 'Windows' in platform.platform() else 32
    _cuda = '1' if socket.gethostname() == 'dell-PowerEdge-T640' else '0'
    _fp16 = 1 if socket.gethostname() == 'dell-PowerEdge-T640' else 0

    parser.add_argument('-e', '--epochs', type=int, default=15, help='upper epoch limit')
    parser.add_argument('-b', '--batch_size', type=int, default=_bs, help='batch size')
    parser.add_argument('-l', '--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('-s', '--lr_schedule', action='store_true', help='set True to use lr scheduler')
    parser.add_argument('-g', '--loss_gamma', type=float, default=0.0, help='focal loss gamma')
    parser.add_argument('--cuda', type=str, default=_cuda, help='cuda visible device id')
    parser.add_argument('--sql', type=int, default=125, help='sequence length')
    parser.add_argument('--report', action='store_true', help='report model and exit')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='verbose level, > 0 means True')
    parser.add_argument('-r', '--resume', action='store_true')
    # parser.add_argument('--bert_name', type=str, default='bert-base-chinese', help="pretrained_model_name_or_path")
    parser.add_argument('--fp16', type=int, default=_fp16, help="whether to use 16-bit float, use 0/1 for false/true")
    # About fp16: https://zhpmatrix.github.io/2019/07/01/model-mix-precision-acceleration/
    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':
    # ========================================================================================== Params & Model
    start_time = time.time()
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device(f'cuda')  # CUDA error: an illegal memory access was encountered
    n_epoch = args.epochs
    batch_size = args.batch_size
    sql = args.sql
    lr = args.lr
    n_label = 17
    grad_norm_clip = 2
    full_finetuning = True  # must be true
    show_progressbar = False
    log(f'\n-[{datetime.now().isoformat()}]==================== \n-Args {str(args)[9:]}')
    train_data_loader, val_data_loader, corpus = get_data_loader('../data/xiaofang', batch_size, sql)

    model: nn.Module
    model = BertZhTokenClassifier_(n_label, p_drop=0.1)  # bert_name=args.bert_name
    model.to(device)
    if args.resume:
        load_last_best(model)
    # ========================================================================================== Train
    if full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:  # only fine-tuning the last FC layer, save more memory
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1) if args.lr_schedule else None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 1e-7) if args.lr_schedule else None
    criterion = cross_entropy_loss_non_pad
    if args.fp16:
        # ref: https://blog.csdn.net/mrjkzhangma/article/details/100704397
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if args.report:
        report_model(log_all_preds=True)
        exit()

    s = 's' if args.lr_schedule else ''
    model_fullname = f"BertZh -bs {batch_size:02d} -lr{s} {lr}".replace('e-0', 'e-')
    dt_now = datetime.now().strftime('%m%d-%H%M')
    pathlib.Path('./logs/csv').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./logs/runs').mkdir(exist_ok=True)
    pathlib.Path('./models').mkdir(exist_ok=True)
    fwriter = open(f'./logs/csv/{model_fullname} ({dt_now}).csv', 'a+')
    swriter = SummaryWriter(log_dir=f'./logs/runs/{model_fullname} ({dt_now}) {socket.gethostname()}')
    print(f'\n(Initializing time: {time.time() - start_time:.1f}s)')

    log(f'[{model_fullname}]')
    clear_prediction_logs()
    start_time = time.time()
    best_val_loss, best_val_history, best_epoch = 1e6, None, -1
    try:
        for epoch in range(1, n_epoch + 1):
            log(f"\n=== Epoch: {epoch}/{n_epoch}\t(lr: {optimizer.param_groups[0]['lr']})")
            train_history = train()
            val_history = evaluate()
            if args.lr_schedule:
                scheduler.step()
            log_and_save()
    except KeyboardInterrupt:
        if epoch > 1:  # at least finished the first epoch
            print('\n=== KeyboardInterrupt, training early stop at epoch: {}/{}\n'.format(epoch, n_epoch))
            epoch = n_epoch
            log_and_save()
        else:
            print('\n=== KeyboardInterrupt, exit\n')

    log(f'\n(Time cost: {get_elapsed_time(start_time)})')
    report_model(val_history=best_val_history)
