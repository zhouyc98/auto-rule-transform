#!/usr/bin/python3.7
#coding=utf-8

from train import *


def test():
    model.eval()
    with torch.no_grad():
        # data_loader = tqdm(test_data_loader) if show_progressbar else test_data_loader
        for inputs, att_mask in test_data_loader:
            # inputs: token_ids, labels: tag_ids
            inputs, att_mask = inputs.to(device), att_mask.to(device)

            outputs = model(inputs, att_mask)  # shape: [bs, sql, n_tags]
            outputs = outputs.view(-1, outputs.shape[-1])  # [bs*sql, n_tags]
            _, predictions = torch.max(outputs, 1)  # return (value, index)

            log_predictions(inputs, None, predictions, 'test', is_test=True, corpus_=corpus)


if __name__ == '__main__':
    args = get_args()
    # device = torch.device('cpu' if 'Windows' in platform.platform() else 'cuda:0')
    device = torch.device('cuda:0')
    batch_size = args.batch_size
    sql = args.sql
    n_label = 17
    full_finetuning = True
    show_progressbar = False
    log(f'\n-[{datetime.now().isoformat()}]==================== \n-Args {str(args)[9:]}')

    test_data_loader, corpus = get_test_data_loader(r'../data/xiaofang', batch_size, sql)

    model: nn.Module
    model = BertZhTokenClassifier_(n_label, p_drop=0.1)
    model.to(device)
    model.load_state_dict(torch.load('./models/_BertZh0_best.pth'))  # todo cuda, amp

    log(f'===== Inference =====')
    start_time = time.time()
    clear_prediction_logs(is_test=True)

    test()

    log('\n#Time cost: ' + get_elapsed_time(start_time))
