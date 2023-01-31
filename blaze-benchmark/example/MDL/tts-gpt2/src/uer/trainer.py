# -*- encoding:utf-8 -*-
import os
import sys
import time
import math
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.data import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.utils.tokenizer import *


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    if args.type_vocab_path:
        type_vocab = Vocab()
        type_vocab.load(args.type_vocab_path)
        args.type_vocab = type_vocab

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)


def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        if args.target == "storylinepropattrpict":

            # Build tokenizer.
            tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

            trainset = OfficialStorylinepropattrpictDataset(args, args.vocab, tokenizer)
            train_sampler = DistributedSampler(trainset, args.world_size, rank)

            '''
            def filter_collate_fn(batch):
                batch = list(filter(lambda x:x[0] is not None, batch))
                if len(batch) == 0: return torch.Tensor()
                return default_collate(batch)

            train_loader = torch.utils.data.DataLoader(trainset, collate_fn=filter_collate_fn, batch_size=args.batch_size,
                sampler=train_sampler, num_workers=args.data_loader_workers)
            '''
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                sampler=train_sampler, num_workers=args.data_loader_workers)
        elif args.target == "storylinepropattrpictmulti":

            # Build tokenizer.
            tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

            trainset = OfficialStorylinepropattrpictmultiDataset(args, args.vocab, tokenizer)
            train_sampler = DistributedSampler(trainset, args.world_size, rank)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                sampler=train_sampler, num_workers=args.data_loader_workers)

            args.all_attr_keys = torch.LongTensor(trainset.all_attr_keys)
            args.all_attr_values = torch.LongTensor(trainset.all_attr_values)

        else:
            if args.encoder == "seq2seq":
                train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, rank, args.world_size, args.seq_length, True)
            else:
                train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        if args.encoder == "seq2seq":
            train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, 0, 1, args.seq_length, True)
        else:
            train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.total_steps*args.warmup, t_total=args.total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    globals().get("train_"+args.target)(args, gpu_id, rank, train_loader, model, optimizer, scheduler)


def train_bert(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct_mlm, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    total_correct_nsp, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    done_tokens = 0
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_mlm, tgt_nsp, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_mlm = tgt_mlm.cuda(gpu_id)
            tgt_nsp = tgt_nsp.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, (tgt_mlm, tgt_nsp), seg)
        loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator = loss_info

         # Backward.
        loss = loss_mlm + loss_nsp
        total_loss += loss.item()
        total_loss_mlm += loss_mlm.item()
        total_loss_nsp += loss_nsp.item()
        total_correct_mlm += correct_mlm.item()
        total_correct_nsp += correct_nsp.item()
        total_denominator += denominator.item()
        total_instances += src.size(0)
        done_tokens += src.size(0) * src.size(1)

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps
            loss_mlm = total_loss_mlm / args.report_steps
            loss_nsp = total_loss_nsp / args.report_steps

            elapsed = time.time() - start_time

            if args.dist_train:
                done_tokens *= args.world_size

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_nsp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_nsp: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    loss_mlm,
                    loss_nsp,
                    total_correct_mlm / total_denominator,
                    total_correct_nsp  / total_instances))

            done_tokens = 0
            total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
            total_correct_mlm, total_denominator = 0., 0.
            total_correct_nsp, total_instances = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_lm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_seq2seq(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg, masks = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            masks = masks.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg, masks)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_storylineprop(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg, prop_keys, prop_values, masks = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            masks = masks.cuda(gpu_id)
            prop_keys = prop_keys.cuda(gpu_id)
            prop_values = prop_values.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg, masks, prop_keys, prop_values)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_storylinepropattr(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg, prop_keys, prop_values, attr_keys, attr_values, masks = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            masks = masks.cuda(gpu_id)
            prop_keys = prop_keys.cuda(gpu_id)
            prop_values = prop_values.cuda(gpu_id)
            attr_keys = attr_keys.cuda(gpu_id)
            attr_values = attr_values.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg, masks, prop_keys, prop_values, attr_keys, attr_values)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_storylinepropattrpict(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = loader

    while True:
        for idx, data in enumerate(loader_iter):
            if steps == total_steps + 1:
                return
            src, tgt, seg, prop_keys, prop_values, attr_keys, attr_values, picts, masks = data

            if gpu_id is not None:
                src = src.cuda(gpu_id)
                tgt = tgt.cuda(gpu_id)
                seg = seg.cuda(gpu_id)
                masks = masks.cuda(gpu_id)
                prop_keys = prop_keys.cuda(gpu_id)
                prop_values = prop_values.cuda(gpu_id)
                attr_keys = attr_keys.cuda(gpu_id)
                attr_values = attr_values.cuda(gpu_id)
                picts = picts.cuda(gpu_id)

            # Forward.
            loss_info = model(src, tgt, seg, masks, prop_keys, prop_values, attr_keys, attr_values, picts)
            loss, correct, denominator = loss_info

            # Backward.
            total_loss += loss.item()
            total_correct += correct.item()
            total_denominator += denominator.item()

            loss = loss / args.accumulation_steps

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if steps % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if steps % args.report_steps == 0  and \
                (not args.dist_train or (args.dist_train and rank == 0)):

                loss = total_loss / args.report_steps

                elapsed = time.time() - start_time

                done_tokens = \
                    args.batch_size * src.size(1) * args.report_steps * args.world_size \
                    if args.dist_train \
                    else args.batch_size * src.size(1) * args.report_steps

                dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("Time::: {} "
                    "| {:8d}/{:8d} steps"
                    "| {:8.2f} tokens/s"
                    "| loss {:7.2f}"
                    "| acc: {:3.3f}".format(
                        dt_ms,
                        steps,
                        total_steps,
                        done_tokens / elapsed,
                        loss,
                        total_correct / total_denominator))

                total_loss = 0.
                total_correct, total_denominator = 0., 0.

                start_time = time.time()

            if steps % args.save_checkpoint_steps == 0 and \
                    (not args.dist_train or (args.dist_train and rank == 0)):
                save_model(model, args.output_model_path + "-" + str(steps))

            steps += 1


def train_storylinepropattrpictmulti(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_lm_loss, total_keys_loss, total_values_loss = 0., 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    total_keys_correct, total_values_correct, total_attr_denominator = 0., 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = loader

    while True:
        for idx, data in enumerate(loader_iter):
            if steps == total_steps + 1:
                return
            #src, tgt, seg, prop_keys, prop_values, attr_keys, attr_values, picts, masks = data
            src, tgt, seg, prop_keys, prop_values, attr_keys, attr_values, attr_keys_target, attr_values_target, picts, masks = data
            all_attr_keys, all_attr_values = args.all_attr_keys, args.all_attr_values

            if gpu_id is not None:
                src = src.cuda(gpu_id)
                tgt = tgt.cuda(gpu_id)
                seg = seg.cuda(gpu_id)
                masks = masks.cuda(gpu_id)
                prop_keys = prop_keys.cuda(gpu_id)
                prop_values = prop_values.cuda(gpu_id)
                attr_keys = attr_keys.cuda(gpu_id)
                attr_values = attr_values.cuda(gpu_id)
                attr_keys_target = attr_keys_target.cuda(gpu_id)
                attr_values_target = attr_values_target.cuda(gpu_id)
                all_attr_keys = all_attr_keys.cuda(gpu_id)
                all_attr_values = all_attr_values.cuda(gpu_id)
                picts = picts.cuda(gpu_id)

            # Forward.
            loss_info = model(src, tgt, seg, masks, prop_keys, prop_values, attr_keys, attr_values, attr_keys_target, attr_values_target, all_attr_keys, all_attr_values, picts)
            lm_loss, correct, denominator, keys_loss, keys_correct, values_loss, values_correct, attr_denominator = loss_info

            loss = (lm_loss + keys_loss + values_loss) / 3

            # Backward.
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_keys_loss += keys_loss.item()
            total_values_loss += values_loss.item()
            total_correct += correct.item()
            total_denominator += denominator.item()
            total_keys_correct += keys_correct.item()
            total_values_correct += values_correct.item()
            total_attr_denominator += attr_denominator.item()

            loss = loss / args.accumulation_steps

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if steps % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if steps % args.report_steps == 0  and \
                (not args.dist_train or (args.dist_train and rank == 0)):

                loss = total_loss / args.report_steps

                elapsed = time.time() - start_time

                done_tokens = \
                    args.batch_size * src.size(1) * args.report_steps * args.world_size \
                    if args.dist_train \
                    else args.batch_size * src.size(1) * args.report_steps

                dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("Time::: {} "
                    "| {:8d}/{:8d} steps"
                    "| {:8.2f} tokens/s"
                    "| lm_loss {:7.2f}"
                    "| lm_acc {:3.3f}"
                    "| keys_loss {:7.2f}"
                    "| keys_acc {:3.3f}"
                    "| values_loss {:7.2f}"
                    "| values_acc {:3.3f}"
                    "| loss {:7.2f}".format(
                        dt_ms,
                        steps,
                        total_steps,
                        done_tokens / elapsed,
                        total_lm_loss / args.report_steps,
                        total_correct / total_denominator,
                        total_keys_loss / args.report_steps,
                        total_keys_correct / total_attr_denominator,
                        total_values_loss / args.report_steps,
                        total_values_correct / total_attr_denominator,
                        loss))

                total_loss, total_lm_loss, total_keys_loss, total_values_loss = 0., 0., 0., 0.
                total_correct, total_denominator = 0., 0.
                total_keys_correct, total_values_correct, total_attr_denominator = 0., 0., 0.

                start_time = time.time()

            if steps % args.save_checkpoint_steps == 0 and \
                    (not args.dist_train or (args.dist_train and rank == 0)):
                save_model(model, args.output_model_path + "-" + str(steps))

            steps += 1


def train_storylinepropcls(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, seg, prop_keys, prop_values, target_words = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            prop_keys = prop_keys.cuda(gpu_id)
            prop_values = prop_values.cuda(gpu_id)
            target_words = target_words.cuda(gpu_id)

        # Forward.
        loss_info = model(src, seg, prop_keys, prop_values, target_words)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_fpdg(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_word_loss, total_type_loss = 0., 0.
    # Calculate MLM accuracy.
    total_word_correct, total_word_denominator = 0., 0.
    total_type_correct, total_type_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, src_type, tgt, tgt_type, seg, masks = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            src_type = src_type.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            tgt_type = tgt_type.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            masks = masks.cuda(gpu_id)

        # Forward.
        loss_info = model(src, src_type, tgt, tgt_type, seg, masks)
        word_loss, word_correct, word_denominator, type_loss, type_correct, type_denominator = loss_info
        loss = word_loss + type_loss
        #loss = word_loss

        # Backward.
        total_word_loss += word_loss.item()
        total_type_loss += type_loss.item()
        total_word_correct += word_correct.item()
        total_type_correct += type_correct.item()
        total_word_denominator += word_denominator.item()
        total_type_denominator += type_denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            word_loss = total_word_loss / args.report_steps
            type_loss = total_type_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| word_loss {:7.2f}"
                  "| type_loss {:7.2f}"
                  "| word_acc: {:3.3f}"
                  "| type_acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    word_loss,
                    type_loss,
                    total_word_correct / total_word_denominator,
                    total_type_correct / total_type_denominator))

            total_word_loss = 0.
            total_type_loss = 0.
            total_word_correct, total_word_denominator = 0., 0.
            total_type_correct, total_type_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_vae(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_kl_loss = 0.
    total_reconstruction_loss = 0.
    total_bow_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    '''
    optimizer_part = AdamW(model.condition_title_mulogvar.parameters(), lr=args.learning_rate, correct_bias=False)
    scheduler_part = WarmupLinearSchedule(optimizer_part, warmup_steps=args.total_steps*args.warmup, t_total=args.total_steps)
    '''

    while True:
        if steps == total_steps + 1:
            break
        condition_title, condition_text, src, tgt, seg, condition_title_seg, condition_text_seg, masks = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
            condition_title = condition_title.cuda(gpu_id)
            condition_title_seg = condition_title_seg.cuda(gpu_id)
            condition_text = condition_text.cuda(gpu_id)
            condition_text_seg = condition_text_seg.cuda(gpu_id)
            masks = masks.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg, condition_title, condition_title_seg, condition_text, condition_text_seg, masks)
        reconstruction_loss, kl_loss, bow_loss, correct, denominator = loss_info

        #kl_weights = 0 if steps <= 20000 else min(float(steps - 20000) / 200000, 1.0)

        #kl_weights = min(float(steps) / 200000, 1.0)
        #g = 0.05

        kl_weights = min((float(steps) % 40000) / 20000, 1.0)
        g = 0.01

        l = 0.5
        #loss = (1 - l) * reconstruction_loss + torch.max(kl_loss, torch.tensor(8.).to(device=kl_loss.device)) * kl_weights * g + bow_loss * l
        loss = (1 - l) * reconstruction_loss + torch.max(kl_loss, torch.tensor(0.).to(device=kl_loss.device)) * kl_weights * g + bow_loss * l

        # Backward.
        total_loss += loss.item()
        total_kl_loss += kl_loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_bow_loss += bow_loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()



        '''
        hidden_size = condition_title_output.size(-1)
        condition_title_mu, condition_title_logvar = model.condition_title_mulogvar(condition_title_output.detach()).split(hidden_size, dim=-1)
        def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
            kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                    - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                    - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), (1, 2))
            return kld

        kl_loss = gaussian_kld(condition_title_mu, condition_title_logvar, condition_text_mu.detach(), condition_text_logvar.detach()).mean()

        total_kl_loss += kl_loss.item()

        kl_loss = kl_loss / args.accumulation_steps

        kl_loss.backward()
        if steps % args.accumulation_steps == 0:
            optimizer_part.step()
            scheduler_part.step()
            optimizer_part.zero_grad()
        '''



        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps
            reconstruction_loss = total_reconstruction_loss / args.report_steps
            kl_loss = total_kl_loss / args.report_steps
            bow_loss = total_bow_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| reconstruction loss {:7.2f}"
                  "| kl loss {:7.2f}"
                  "| bow loss {:7.2f}"
                  "| total loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    reconstruction_loss,
                    kl_loss,
                    bow_loss,
                    loss,
                    total_correct / total_denominator))

            total_reconstruction_loss = 0.
            total_kl_loss = 0.
            total_bow_loss = 0.
            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_bilm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
    # Calculate BiLM accuracy.
    total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_forward, tgt_backward, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_forward = tgt_forward.cuda(gpu_id)
            tgt_backward = tgt_backward.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info

        # Backward.
        loss = loss_forward + loss_backward
        total_loss += loss.item()
        total_loss_forward += loss_forward.item()
        total_loss_backward += loss_backward.item()
        total_correct_forward += correct_forward.item()
        total_correct_backward += correct_backward.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    loss_forward,
                    loss_backward,
                    total_correct_forward / total_denominator,
                    total_correct_backward / total_denominator))

            total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
            total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_cls(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_correct, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_instances += src.size(0)

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_instances))

            total_loss = 0.
            total_correct = 0.
            total_instances = 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_mlm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    done_tokens / elapsed,
                    loss,
                    total_correct / total_denominator))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


# def train_nsp(args, gpu_id, rank, loader, model, optimizer):
#     model.train()
#     start_time = time.time()
#     total_loss = 0.
#     total_correct, total_instances = 0., 0.
#     steps = 1
#     total_steps = args.total_steps
#     loader_iter = iter(loader)

#     while True:
#         if steps == total_steps + 1:
#             break
#         src, tgt, seg = next(loader_iter)

#         if gpu_id is not None:
#             src = src.cuda(gpu_id)
#             tgt = tgt.cuda(gpu_id)
#             seg = seg.cuda(gpu_id)

#         # Forward.
#         loss_info = model(src, tgt, seg)
#         loss, correct = loss_info

#         # Backward.
#         total_loss += loss.item()
#         total_correct += correct.item()
#         total_instances += src.size(0)

#         loss = loss / args.accumulation_steps
#         loss.backward()

#         if steps % args.accumulation_steps == 0:
#             optimizer.step()
#             model.zero_grad()

#         if steps % args.report_steps == 0  and \
#             (not args.dist_train or (args.dist_train and rank == 0)):

#             loss = total_loss / args.report_steps

#             elapsed = time.time() - start_time

#             done_tokens = \
#                 args.batch_size * src.size(1) * args.report_steps * args.world_size \
#                 if args.dist_train \
#                 else args.batch_size * src.size(1) * args.report_steps

#             print("| {:8d}/{:8d} steps"
#                   "| {:8.2f} tokens/s"
#                   "| loss {:7.2f}"
#                   "| acc: {:3.3f}".format(
#                     steps,
#                     total_steps,
#                     done_tokens / elapsed,
#                     loss,
#                     total_correct / total_instances))

#             total_loss = 0.
#             total_correct = 0.
#             total_instances = 0.

#             start_time = time.time()

#         if steps % args.save_checkpoint_steps == 0 and \
#                 (not args.dist_train or (args.dist_train and rank == 0)):
#             save_model(model, args.output_model_path + "-" + str(steps))

#         steps += 1


# def train_s2s(args, gpu_id, rank, loader, model, optimizer):
#     model.train()
#     start_time = time.time()
#     total_loss= 0.
#     total_correct, total_denominator = 0., 0.
#     steps = 1
#     total_steps = args.total_steps
#     loader_iter = iter(loader)

#     while True:
#         if steps == total_steps + 1:
#             break
#         src, tgt, seg = next(loader_iter)

#         if gpu_id is not None:
#             src = src.cuda(gpu_id)
#             tgt = tgt.cuda(gpu_id)
#             seg = seg.cuda(gpu_id)

#         # Forward.
#         loss_info = model(src, tgt, seg)
#         loss, correct, denominator = loss_info

#         # Backward.
#         total_loss += loss.item()
#         total_correct += correct.item()
#         total_denominator += denominator.item()

#         loss = loss / args.accumulation_steps
#         loss.backward()

#         if steps % args.accumulation_steps == 0:
#             optimizer.step()
#             model.zero_grad()

#         if steps % args.report_steps == 0  and \
#             (not args.dist_train or (args.dist_train and rank == 0)):

#             loss = total_loss / args.report_steps

#             elapsed = time.time() - start_time

#             done_tokens = \
#                 args.batch_size * src.size(1) * args.report_steps * args.world_size \
#                 if args.dist_train \
#                 else args.batch_size * src.size(1) * args.report_steps

#             print("| {:8d}/{:8d} steps"
#                   "| {:8.2f} tokens/s"
#                   "| loss {:7.2f}"
#                   "| acc: {:3.3f}".format(
#                     steps,
#                     total_steps,
#                     done_tokens / elapsed,
#                     loss,
#                     total_correct / total_denominator))

#             total_loss = 0.
#             total_correct, total_denominator = 0., 0.

#             start_time = time.time()

#         if steps % args.save_checkpoint_steps == 0 and \
#                 (not args.dist_train or (args.dist_train and rank == 0)):
#             save_model(model, args.output_model_path + "-" + str(steps))

#         steps += 1
