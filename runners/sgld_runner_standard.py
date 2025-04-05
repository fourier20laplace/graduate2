import importlib
import os, sys
import copy
import random
import numpy as np
import time

import torch
from torch import nn

from utils.data_utils      import *
from utils.optimizer_utils import get_optimizer

from tools.evaluator       import Evaluator
from tools.intervention    import ImageIntervention

from models.compressor_bptt_standard import CompressorBPTT



"""
  Set up the interventions
"""
def set_up_interventions(config):
    if hasattr(config, 'intervention') and config.intervention.strategy != 'none':
        test_intervention  = ImageIntervention(
                                 config.intervention.test_name,
                                 config.intervention.test_strategy,
                                 phase='test',
                             )
        train_intervention = ImageIntervention(
                                 config.intervention.train_name,
                                 config.intervention.train_strategy,
                                 phase='train',
                             )
        # This is a customizable prob \in [0, 1]
        intervention_prob  = 1.0
    else:
        test_intervention  = None
        train_intervention = None
        intervention_prob  = 0

    return train_intervention, test_intervention, intervention_prob


"""
  Sample a subset of tasks (indices)
"""
def get_task_indices(train_n_classes, task_sampler_nc):
    task_indices = list(range(train_n_classes))
    if task_sampler_nc > 0:
        random.shuffle(task_indices)
        task_indices = task_indices[:task_sampler_nc]
        task_indices.sort()
    return task_indices


"""
  Assign grads to a network
"""
def assign_grads(model, grads):
    assert isinstance(grads, list) or isinstance(grads, tuple)
    params = list(model.parameters())
    for p, g in zip(params, grads):
       p.grad = g


"""
  Flatten tensors
"""
def flatten(data):
    return torch.cat([ele.flatten() for ele in data])


"""
  Get generalization data batch
"""
def get_generalization(compressor_dset, task_indices, n_queries=64):
    #? 可能需要改一下 n_queries
    # CIFAR10每个类别6000张 但我的resisc45 训练集每个类别只有400张左右
    # 选64倒也行吧 
    # n_queries=32
    n_queries=128
    data_x, data_labels = compressor_dset(task_indices=task_indices, new_batch_size=n_queries)
    return [data_x, data_labels]


"""
  Train function for compressor
"""
def train(rank, world_size, config, args, writer,save_path):
    print(f"Running compressor training on rank {rank}.")

    eval_intervals = np.arange(0, config.training.n_iters + 1, config.training.eval_every)

    eval_models    = [config.backbone.name]
    if hasattr(config, 'cross_archs'):
        eval_models = config.cross_archs.name.split(",")

    # import compressor libraries
    compressor_lib      = importlib.import_module('compressors.' + config.compressor.name)
    compressor_dset_lib = importlib.import_module('compressors.dset')

    # set up interventions
    train_intervention, test_intervention, intervention_prob = set_up_interventions(config)
#应该是pair_aug的增强？
    # prepare dataset
    # channel, im_size, train_n_classes, test_n_classes, dst_train, dst_test = get_dataset(
    #     config.dataset.name,
    #     config.dataset.data_path,
    #     zca=args.zca==1
    # )
    channel, im_size, train_n_classes, test_n_classes, dst_train, dst_test = get_dataset(
        config.dataset.name,
        "/home/lmh/.cache/huggingface/hub/datasets--timm--resisc45",
        zca=args.zca==1
    )
    # train_images_all, train_labels_all, train_indices_class = organize_dst(
    #                                                               dst_train,
    #                                                               train_n_classes,
    #                                                               print_info=rank==0
    #                                                           )
    #! 使用自己的组织方式
    train_images_all, train_labels_all, train_indices_class = my_organize_dst(
                                                                  dst_train,
                                                                  train_n_classes,
                                                                  debug=False, #说来话长但是这个地方如果只是简单的取前几个的话 后面选取task_indices可能导致空
                                                                  print_info=False
                                                              )
    # test_images_all,  test_labels_all,  test_indices_class  = organize_dst(
    #                                                               dst_test,
    #                                                               test_n_classes,
    #                                                               print_info=rank==0
    #                                                           )
    if args.debug:
        test_loader = torch.utils.data.DataLoader(dst_test.select(range(32)), batch_size=4, shuffle=False, num_workers=2)
        config.evaluation.train_epoch = 4
        config.evaluation.num_eval = 3 # 代码会多次评估取均值和标准差
    else:
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)

    args.task_sampler_nc = min(args.task_sampler_nc, train_n_classes)


    """
      Define compressor model, bptt model, optimizer, and dset
          - compressor
          - compressor_bptt
          - compressor optimizer
          - compressor_dset
    """
    compressor  = compressor_lib.Net(
                      config.compressor.ipc,
                      channel,
                      im_size,
                      train_n_classes,
                      rank,
                      downsample_scale=config.compressor.downsample_scale,
                      n_basis=args.n_basis,
                      n_per_c=args.compressor_minibatch_size,
                  )

    compressor_bptt = CompressorBPTT(
                          compressor,
                          config,
                          (channel, im_size),
                          train_intervention,
                          backbone_ways=train_n_classes,
                          coeff_reg=args.coeff_reg,
                          coeff_reg_alpha=args.coeff_reg_alpha,
                      ).to(rank)

    optimizer_compressor = get_optimizer(compressor_bptt.compressor.parameters(), config.compressor_optim)

    print('Compressor optimizer:', optimizer_compressor)

    compressor_dset = compressor_dset_lib.Net(
                          (train_images_all, train_indices_class),
                          train_n_classes,
                          rank,
                          batch_size=args.dset_batch_size,
                      )

    # Define evaluators
    evaluator = Evaluator(
                    config,
                    (channel, test_n_classes, im_size),
                    eval_models,
                    test_loader=test_loader,
                    rank=rank,
                )

    sys.stdout.flush()

    # Train for N iterations
    new_lr = optimizer_compressor.param_groups[0]['lr']
    for it in range(config.training.n_iters):
        # evaluate at intervals
        myEval=True
        if rank == 0 and it in eval_intervals and myEval:
            # if hasattr(config, 'save') and it % config.save.save_every==0:
            #     save(compressor)
            print('Evaluating... Learning rate: ', new_lr, 'Iteration:', it)
            print('Downsample scale: ', compressor.downsample_scale)
            compressor_set = [copy.deepcopy(compressor_bptt.compressor)] #可以看到这是一个列表 作者的本意应该是多个compressor的测试？

            for _train_epoch, _prob in [(config.evaluation.train_epoch, intervention_prob)]:
                # [(config.evaluation.train_epoch, intervention_prob)] 是一个列表，其中有一个元组。
                # 这个列表只有一个元素，该元素是一个二元元组。
                # 在 for 循环中，_train_epoch, _prob 是两个变量，它们会分别接收元组中的第一个和第二个元素。
                print(
                    f"Evaluating using {_prob} prob interventions... Current learning rate: ",\
                    new_lr, 'Current iteration:', it
                )
                print('Current downsample scale: ', compressor_bptt.compressor.downsample_scale)
                
                #! 为了集成tb 做了一定的修改
                # evaluator.evaluate(
                #     compressor_set,
                #     current_iter=it,
                #     train_epoch=_train_epoch,
                #     num_eval=config.evaluation.num_eval,
                #     intervention=[test_intervention, _prob]
                # )
                test_acc, std = evaluator.my_evaluate(
                    compressor_set,
                    current_iter=it,
                    train_epoch=_train_epoch,
                    num_eval=config.evaluation.num_eval,
                    intervention=[test_intervention, _prob]
                )
                writer.add_scalar('val/mean_acc', test_acc, it)
                writer.add_scalar('val/std_acc', std, it)
                my_save(compressor,it,test_acc,std,save_path)
        # Get data for generalization batch loss
        task_indices        = get_task_indices(train_n_classes, args.task_sampler_nc) #只取args.task_sampler_nc个task来算generalization loss
        generalization_data = get_generalization(compressor_dset, task_indices)

        # Optimize compressor_bptt model with inner loops
        optimizer_compressor.zero_grad()
        loss, dL_dc, dL_dw  = compressor_bptt.forward(
                                  task_indices=task_indices,
                                  generalization_data=generalization_data,
                                  intervention_seed=int(time.time() * 1000) % 100000,
                              )
        torch.cuda.empty_cache()

        if isinstance(dL_dc, list):
            compressor.assign_grads([flatten(ele) for ele in dL_dc], task_indices=task_indices)
        else:
            # compressor.assign_grads(flatten(dL_dc), task_indices=task_indices)
            #! renyouduodadan diyouduodachan
            myIMGS=compressor_bptt.compressor()[0]
            # myIMGS.zero_grad()
            # myIMGS.grad=dL_dc
            myIMGS.backward(dL_dc)

        torch.nn.utils.clip_grad_norm_(compressor_bptt.parameters(), max_norm=2)
        optimizer_compressor.step()
        optimizer_compressor.zero_grad()

        # Logging
        if it % config.training.print_every == 0 and rank == 0:
            print('it', it, 'train loss', loss)
            print('compressor min max: ', compressor.get_min_max())
            
        #todo: 为了观察 把writer放在外面了 即每个it都写入
        writer.add_scalar('train/loss', loss, it)

        sys.stdout.flush()

    print('Training completed.')


"""
  Train compressor for multiple runs
"""
def train_multi_run(config, args):
    from torch.utils.tensorboard import SummaryWriter
    
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    if args.debug:
        log_root="/home/lmh/projects_dir/graduate/RememberThePast-DatasetDistillation/logs/tensorboard/debug"
        save_root="/home/lmh/projects_dir/graduate/RememberThePast-DatasetDistillation/logs/saves/debug"
    else:
        log_root="/home/lmh/projects_dir/graduate/RememberThePast-DatasetDistillation/logs/tensorboard"
        save_root="/home/lmh/projects_dir/graduate/RememberThePast-DatasetDistillation/logs/saves"
    log_path=os.path.join(log_root,timestamp)
    save_path=os.path.join(save_root,timestamp)
    os.makedirs(log_path,exist_ok=True)
    writer=SummaryWriter(log_path)
    os.makedirs(save_path,exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print('world size:', world_size)
    for exp in range(config.training.num_exp):
        print(('\n' + '='*18 + ' Exp %d ' + '='*18 + '\n ') % exp)
        train(0, world_size, config, args, writer,save_path)


"""
  Save compressor state dict
"""
def save(compressor, dataset):
    print("saving compressor")
    torch.save(
        {'compressor': compressor.state_dict()},
        'saves/' + compressor.name + '_' + dataset + '_compressor.pt'
    )
def my_save(compressor,it,test_acc,std,save_path):
    print("saving compressor")
    torch.save(
        {'compressor': compressor.state_dict()},
        os.path.join(save_path,compressor.name+'_'+str(it) +'_'+str(test_acc)+'_'+str(std)+'.pt')
    )
