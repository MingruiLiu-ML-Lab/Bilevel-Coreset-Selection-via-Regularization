import time
from comet_ml import Experiment
import os
import torch
import numpy as np
import json
import argparse
from core.data_utils import get_all_loaders
from core.train_methods_cifar import train_task_sequentially, eval_single_epoch
from core.utils import setup_experiment, get_random_string
from core.utils import save_task_model_by_policy
import random as rnd
import datetime

def get_arguments():
    """
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for split cifar experiment.")
    parser.add_argument("--exp_name", type=str, default='cl')
    parser.add_argument("--select_type", type=str, default='bcsr',  help="method name")
    parser.add_argument("--seed", type=int, default=0,  help="random seed")
    parser.add_argument("--trial", type=str, default=None,  help="")
    parser.add_argument("--exp_dir", type=str, default=None,  help="")
    parser.add_argument("--dataset", type=str, default='cifar-bs-50',  help="dataset name")
    parser.add_argument("--device", type=str, default='cuda',  help="training device cpu or cuda")
    # model hyperparameters
    parser.add_argument("--mlp_hiddens", type=int, default=256,  help="number of hidden unit")
    parser.add_argument("--seq_lr", type=float, default=0.15,  help="learning rate for training")
    parser.add_argument("--lr_decay", type=float, default=0.9,  help="learning rate decay")
    parser.add_argument("--lr_proxy_model", type=float, default=5.0,  help="lr for model for coreset selection")
    parser.add_argument("--lr_weight", type=float, default=5.0,  help="lr for sample weights")
    parser.add_argument("--momentum", type=float, default=0.8,  help="learning momentum")
    parser.add_argument("--dropout", type=float, default=0.2,  help="drop out rate")
    parser.add_argument("--memory_size", type=int, default=100,  help="size of buffer data")
    parser.add_argument("--stream_size", type=int, default=50,  help="size of stream mini-batch")
    parser.add_argument("--batch_size", type=int, default=10,  help="size of training data")
    parser.add_argument("--n_classes", type=int, default=5,  help="number of classes")
    parser.add_argument("--ref_hyp", type=float, default=0.5,  help="coefficient for balancing the current loss and reference loss")
    parser.add_argument("--beta", type=float, default=0.1,  help="coefficient for balancing the current loss and regularizer")
    parser.add_argument("--num_tasks", type=int, default=20,  help="number of tasks in continual learning")
    parser.add_argument("--seq_epochs", type=int, default=1,  help="epochs for continual learning")
    parser.add_argument("--outer_iter", type=int, default=5,  help="optimization iterations for outer loops")
    parser.add_argument("--inner_iter", type=int, default=1,  help="optimization iterations for inner loops")

    return parser.parse_args()


def main():
    # DATASET = 'cifar-bs-50' or 'imb-cifar-bs-50'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRIAL_ID = os.environ.get('NNI_TRIAL_JOB_ID', get_random_string(5))
    EXP_DIR = './checkpoints/{}'.format(TRIAL_ID)

    coreset_methods = ['uniform', 'coreset', 'bcsr']

    # Get the CL arguments
    args = get_arguments()

    if 'imb-cifar' or 'noise-cifar' in args.dataset:
        args.lr_decay = 0.875
        args.ref_hyp = 0.1
        args.lr_proxy_model = 10
        args.lr_weight = 10

    now = datetime.datetime.now()
    Thistime = now.strftime('%Y-%m-%d-%H-%M-%S')
    print(Thistime)
    args.exp_dir = EXP_DIR
    seed = args.seed
    CUDA_VISIBLE_DEVICES = 0
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    args.exp_name = Thistime + '-' + args.dataset + '-' + args.select_type + '_seqlr' + (
            '%s' % args.seq_lr).replace('.', 'p') + '_seqep' + '%s' % args.seq_epochs + '_seqbs' + '%s' % \
                        args.stream_size


    args.trial = TRIAL_ID
    experiment = Experiment(api_key="hidden_key", \
                            project_name="rot-mnist-20", \
                            workspace="cl-modeconnectivity", disabled=True)

    loaders = get_all_loaders(seed, args.dataset, args.num_tasks, \
                              args.batch_size, args.stream_size, \
                              args.memory_size)

    print(args)

    setup_experiment(experiment, args)
    st = time.time()
    accs_max = [0. for _ in range(args.num_tasks)]
    accs_avg = [0. for _ in range(args.num_tasks)]
    outer_loss = []

    for task in range(1, args.num_tasks+1):
        accs_max_temp = [0. for _ in range(args.num_tasks)]
        print('---- Task {} ----'.format(task))
        seq_model, outer_loss = train_task_sequentially(args, task, loaders, outer_loss)
        save_task_model_by_policy(seq_model, task, 'seq', args.exp_dir)
        if not os.path.exists('./data/loss_data/'):
            os.makedirs('./data/loss_data/')
        torch.save(outer_loss, './data/loss_data/'+args.exp_name+'.pkl')

        accs_rcp_temp, losses_rcp_temp = [], []
        for prev_task in range(1, task+1):
            metrics_rcp = eval_single_epoch(seq_model, loaders['sequential'][prev_task]['val'], args)
            accs_rcp_temp.append(metrics_rcp['accuracy'])
            losses_rcp_temp.append(metrics_rcp['loss'])
            print('>>> ', prev_task, metrics_rcp)
            accs_max_temp[prev_task-1] = metrics_rcp['accuracy'] if accs_max[prev_task-1] < metrics_rcp['accuracy'] else accs_max[prev_task-1]

        print("Average accuracy: {}".format(np.mean(accs_rcp_temp)))
        print("Forgetting: {}".format(np.sum(np.array(accs_max[:task-1])-np.array(accs_rcp_temp[:task-1]))/(task-1)))
        accs_avg[task-1] = np.mean(accs_rcp_temp)
        print('avg per-task accuracy >>): {}'.format(accs_avg))
        accs_max = accs_max_temp
        et = time.time()
        total_time = (et - st) / 3600
        with open('summary/' + args.exp_name, 'w') as outfile:
            json.dump({'Exp configuration': str(args), 'AVG ACC': np.mean(accs_rcp_temp), 'Forgetting': np.sum(np.array(accs_max[:task-1])-np.array(accs_rcp_temp[:task-1]))/(task-1),
                       'per-task accuracy': accs_max, 'acc_avg': accs_avg, 'time': total_time}, outfile)
        print('total time:', total_time)

    print(args)
    experiment.end()

if __name__ == "__main__":
    main()
