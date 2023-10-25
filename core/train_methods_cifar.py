import pdb
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributions as td

from .utils import DEVICE, save_model,load_model
from .utils import flatten_params, flatten_grads, flatten_example_grads, assign_weights, assign_grads, accum_grads, compute_and_flatten_example_grads
from .mode_connectivity import get_line_loss, get_coreset_loss, reconstruct_coreset_loader2
from .summary import Summarizer
from .bilevel_coreset import BilevelCoreset
from .bcsr_coreset import BCSR_Coreset
from .ntk_generator import generate_fnn_ntk, generate_cnn_ntk, generate_resnet_ntk
from .models import ResNet18, MLP
import copy

gumbel_dist = td.gumbel.Gumbel(0,1)


def get_kernel_fn():
        return lambda x, y: generate_resnet_ntk(x.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1))

def coreset_cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value

kernel_fn = get_kernel_fn()

coreset_methods = ['uniform', 'coreset', 'bcsr']

def classwise_fair_selection(task, cand_target, sorted_index, num_per_label, args, is_shuffle=True):
    num_examples_per_task = args.memory_size // task
    num_examples_per_class = num_examples_per_task // args.n_classes
    num_residuals = num_examples_per_task - num_examples_per_class * args.n_classes
    residuals =  np.sum([(num_examples_per_class - n_c)*(num_examples_per_class > n_c) for n_c in num_per_label])
    num_residuals += residuals
    # Get the number of coreset instances per class
    while True:
        n_less_sample_class =  np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
        num_class = (args.n_classes-n_less_sample_class)
        if (num_residuals // num_class) > 0:
            num_examples_per_class += (num_residuals // num_class)
            num_residuals -= (num_residuals // num_class) * num_class
        else:
            break
    # Get best coresets per class
    selected = []
    target_tid = np.floor(max(cand_target)/args.n_classes)

    for j in range(args.n_classes):
        position = np.squeeze((cand_target[sorted_index]==j+(target_tid*args.n_classes)).nonzero())
        if position.numel() > 1:
            selected.append(position[:num_examples_per_class])
        elif position.numel() == 0:
            continue
        else:
            selected.append([position])
    # Fill rest space as best residuals
    selected = np.concatenate(selected)
    unselected = np.array(list(set(np.arange(num_examples_per_task))^set(selected)))
    final_num_residuals = num_examples_per_task - len(selected)
    best_residuals = unselected[:final_num_residuals]
    selected = np.concatenate([selected, best_residuals])

    if is_shuffle:
        random.shuffle(selected)

    return sorted_index[selected.astype(int)]


def select_coreset(loader, task, model, candidates, args, candidate_size=1000, fair_selection=True, bc=None):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    temp_optimizer = torch.optim.SGD(model.parameters(), lr=args.seq_lr, momentum=args.momentum)
    temp_optimizer.zero_grad()
    print('**************select coreset******************')
    if fair_selection:
        # collect candidates
        cand_data, cand_target = [], []
        cand_size = len(candidates)
        for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
            if batch_idx == cand_size:
                break
            try:
                cand_data.append(data[candidates[batch_idx]])
                cand_target.append(target[candidates[batch_idx]])
            except IndexError:
                pass
        cand_data = torch.cat(cand_data, 0)
        cand_target = torch.cat(cand_target, 0)

        random_pick_up = torch.randperm(len(cand_target))[:candidate_size]
        cand_data = cand_data[random_pick_up]
        cand_target = cand_target[random_pick_up]

        num_per_label = [len((cand_target==(jj+args.n_classes*(task-1))).nonzero()) for jj in range(args.n_classes)]

        num_examples_per_task = args.memory_size // task

        rs = np.random.RandomState(0)

        if args.select_type == 'bcsr':
            size = num_examples_per_task
            pick, _ = bc.coreset_select(model, cand_data.cpu().numpy(), cand_target.cpu().numpy(), task,
                                          topk=len(cand_target))
            pick = classwise_fair_selection(task, cand_target, pick, num_per_label, args, is_shuffle=True)

        elif args.select_type == 'coreset':
            pick, _, = bc.build_with_representer_proxy_batch(cand_data.cpu().numpy(), cand_target.cpu().numpy(),
                                                                 len(cand_target), kernel_fn,
                                                                 data_weights=None, cache_kernel=True,
                                                                 start_size=1, inner_reg=1e-3)
            pick = classwise_fair_selection(task, cand_target, pick, num_per_label, args, is_shuffle=True)
        else:
            summarizer = Summarizer.factory(args.select_type, rs)
            pick = summarizer.build_summary(cand_data.cpu().numpy(), cand_target.cpu().numpy(), num_examples_per_task, task_id=task, method=args.select_type, model=model, device=DEVICE)

        loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[pick])
        loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[pick])

    else:
        pass

def update_coreset(loader, task, model, args, bc=None):
    # Coreset update
    num_examples_per_task = args.memory_size // task
    print('**************update coreset******************')
    for tid in range(1, task):

        if args.select_type in coreset_methods:
            tid_coreset = loader['coreset'][tid]['train'].data
            tid_targets = loader['coreset'][tid]['train'].targets
            num_per_label = [len((tid_targets.cpu()==jj).nonzero()) for jj in range(args.n_classes)]
            rs = np.random.RandomState(0)
            if args.select_type != 'coreset' and args.select_type != 'bcsr':
                summarizer = Summarizer.factory(args.select_type, rs)
                if len(loader['coreset'][tid]['train'].targets.cpu().numpy()) <= num_examples_per_task:
                    selected = np.arange(0, num_examples_per_task)
                else:
                    selected = summarizer.build_summary(loader['coreset'][tid]['train'].data.cpu().numpy(),
                                                        loader['coreset'][tid]['train'].targets.cpu().numpy(),
                                                        num_examples_per_task, task_id=tid, method=args.select_type, model=model, device=DEVICE, taskid=tid)
            elif args.select_type == 'coreset':
                pick, _,= bc.build_with_representer_proxy_batch(tid_coreset.cpu().numpy(), tid_targets.cpu().numpy(),
                                                                 len(tid_targets), kernel_fn,
                                                                 data_weights=None, cache_kernel=True,
                                                                 start_size=1, inner_reg=1e-3)
                selected = classwise_fair_selection(task, tid_targets, pick, num_per_label, args)
            elif args.select_type == 'bcsr':
                pick = bc.outer_loss(tid_coreset.cpu().numpy(), tid_targets.cpu().numpy(), task,
                                             topk=len(tid_targets))
                _,pick = pick.sort(descending=False)
                selected = classwise_fair_selection(task, tid_targets, pick, num_per_label, args)
            else:
                selected = np.arange(0, num_examples_per_task)

        loader['coreset'][tid]['train'].data = copy.deepcopy(loader['coreset'][tid]['train'].data[selected])
        loader['coreset'][tid]['train'].targets = copy.deepcopy(loader['coreset'][tid]['train'].targets[selected])

def train_single_step(model, our_bc, optimizer, loader, task, step, outer_loss, args):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == args.n_substeps else False
    rs = np.random.RandomState(0)
    if args.select_type == 'uniform':
        summarizer = Summarizer.factory(args.select_type, rs)

    candidates_indices=[]
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()

        if args.select_type in coreset_methods:
            size = min(len(data), args.batch_size)
            pick = torch.randperm(len(data))[:size]
            if is_last_step:
                if args.select_type == 'coreset':
                    if len(data) > args.batch_size:
                        pick, _, outer_loss = our_bc.build_with_representer_proxy_batch(data.cpu().numpy(), target.cpu().numpy(),
                                                                                  args.batch_size, kernel_fn, outer_loss,
                                                                                  data_weights=None, cache_kernel=True,
                                                                                  start_size=1, inner_reg=1e-3)
                elif args.select_type == 'bcsr':
                    if len(data) > args.batch_size:
                        pick, outer_loss = our_bc.coreset_select(model, data.cpu().numpy(), target.cpu().numpy(), task_id,
                                                                              topk=args.batch_size, out_loss=outer_loss)
                else:
                    if len(data) > args.batch_size:
                        pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(),
                                                            args.batch_size, task_id=task,
                                                            method=args.select_type, model=model, device=DEVICE,
                                                            taskid=task_id)

            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()
            if is_last_step:
                candidates_indices.append(pick)
        else:
            size = min(len(data), args.batch_size)
            pick = torch.randperm(len(data))[:size]
            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()
        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, args, bc=our_bc)
    return model, outer_loss


def train_coreset_single_step(model, our_bc, optimizer, loader, task, step, outer_loss, args):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == args.n_substeps else False

    ref_loader = reconstruct_coreset_loader2(args, loader['coreset'], task-1)
    ref_iterloader = iter(ref_loader)
    rs = np.random.RandomState(0)
    if args.select_type != 'coreset' and args.select_type != 'bcsr':
        summarizer = Summarizer.factory(args.select_type, rs)
    candidates_indices=[]
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        optimizer.zero_grad()
        is_rand_start = True if (step == 1) else False
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        size = min(args.batch_size,len(data))
        if is_rand_start:
            pick = torch.randperm(len(data))[:size]
        elif len(data) > args.batch_size and args.select_type=='bcsr':
            pick, outer_loss = our_bc.coreset_select(model, data.cpu().numpy(), target.cpu().numpy(), task_id,
                                              args.batch_size, outer_loss, None, None)

        elif len(data) > args.batch_size and args.select_type == 'coreset':
            pick, _, outer_loss = our_bc.build_with_representer_proxy_batch(data.cpu().numpy(), target.cpu().numpy(),
                                                                     args.batch_size, kernel_fn, outer_loss,
                                                                     data_weights=None, cache_kernel=True,
                                                                     start_size=1, inner_reg=1e-3)
        else:
            pick = torch.randperm(len(data))[:size]
        pred = model(data[pick], task_id)
        loss = criterion(pred, target[pick])
        try:
            ref_data = next(ref_iterloader)
        except StopIteration:
            ref_iterloader = iter(ref_loader)
            ref_data = next(ref_iterloader)
        ref_loss = get_coreset_loss(model, ref_data)
        loss += args.ref_hyp * ref_loss
        loss.backward()

        if is_last_step:
            if len(data) > args.batch_size and args.select_type !='bcsr' and args.select_type !='coreset':
                pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(), args.batch_size,
                                                    method=args.select_type, model=model, device=DEVICE,
                                                    task_id=task_id)

            candidates_indices.append(pick)
        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, args, bc=our_bc)
        update_coreset(loader, task_id, model, args, bc=our_bc)
    return model, outer_loss

def eval_single_epoch(net, loader, args):
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    class_correct = [0 for _ in range(args.n_classes)]
    class_total = [0 for _ in range(args.n_classes)]


    with torch.no_grad():
        for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            count += len(target)
            output = net(data, task_id)

            test_loss += criterion(output, target).item()*len(target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            correct_bool = pred.eq(target.data.view_as(pred))

            for cid in range(args.n_classes):
                cid_index = torch.where(target==(cid+(task_id-1)*args.n_classes), torch.ones_like(target), torch.zeros_like(target))
                class_correct[cid] += (cid_index.data.view_as(correct_bool) * correct_bool).sum().item()
                class_total[cid] += cid_index.sum().item()
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count

    pc_avg_acc = [np.round(a/(b+1e-10), 4) for a,b in zip(class_correct, class_total)]
    return {'accuracy': avg_acc, 'per_class_accuracy':pc_avg_acc, 'loss': test_loss}


def train_task_sequentially(args, task, train_loader, outer_loss):
    EXP_DIR = args.exp_dir
    current_lr = args.seq_lr * (args.lr_decay)**(task-1)
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)
    bc = None
    if args.select_type == 'bcsr':
        config = {'n_classes':args.n_classes, 'dropout':args.dropout, 'mlp_hiddens':args.mlp_hiddens}
        proxy_model = ResNet18(config=config).cuda()
        bc = BCSR_Coreset(proxy_model, args.lr_proxy_model, args.beta, out_dim=100,
                              max_outer_it=args.outer_iter, max_inner_it=args.inner_iter,
                              weight_lr=args.lr_weight, candidate_batch_size=600, logging_period=1000)
    if args.select_type == 'coreset':
        def coreset_cross_entropy(K, alpha, y, weights, lmbda):
            loss = torch.nn.CrossEntropyLoss(reduction='none')
            loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
            if lmbda > 0:
                loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
            return loss_value

        bc = BilevelCoreset(outer_loss_fn=coreset_cross_entropy,
                                inner_loss_fn=coreset_cross_entropy, out_dim=100, max_outer_it=args.outer_iter,
                                candidate_batch_size=600, max_inner_it=300, logging_period=10)

    args.n_substeps = int(args.seq_epochs * (args.stream_size / args.batch_size))

    for _step in range(1, args.n_substeps+1):
        if task > 1:
            model, outer_loss = train_coreset_single_step(model, bc, optimizer, train_loader, task, _step, outer_loss, args)
        elif task == 1:
            model, outer_loss = train_single_step(model, bc, optimizer, train_loader, task, _step, outer_loss, args)
        else:
            pass
        metrics = eval_single_epoch(model, train_loader['sequential'][task]['val'], args)
        print('Epoch {} >> (per-task accuracy): {}'.format(_step/args.n_substeps, np.mean(metrics['accuracy'])))
        print('Epoch {} >> (class accuracy): {}'.format(_step/args.n_substeps, metrics['per_class_accuracy']))
    return model, outer_loss
