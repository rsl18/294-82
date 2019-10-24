import argparse
import os
import setproctitle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from segmentron.data import DATASETS, PREPROCS, prepare_data, prepare_loader
from segmentron.model import ARCHS, BACKBONES, HEADS, prepare_model
from segmentron.metrics import LossMeter
from segmentron.options import prepare_parser
from segmentron.util import setup_output, setup_logging

from evaluate import evaluate


def main(args):
    setproctitle.setproctitle(f"segtrain-{args.output}")
    output_dir=f'{args.output}/{args.user}/{args.timestamp}'
    setup_output(output_dir)
    log = setup_logging(f'{output_dir}/log')
    log.info(f"args: {vars(args)}")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = prepare_data(args.dataset, args.split, args.preproc)
    loader = prepare_loader(dataset)

    do_eval = args.eval_dataset is not None
    if do_eval:
        eval_dataset = prepare_data(args.eval_dataset, args.eval_split,
                                    args.preproc)
        eval_loader = prepare_loader(eval_dataset, evaluation=True)

    segmentor = prepare_model(args.arch, args.backbone, args.head,
                              dataset.num_classes, args.weights).to(device)
    segmentor.train()

    loss_fn = nn.CrossEntropyLoss(reduction='sum',
            ignore_index=dataset.ignore_index)
    learned_params = filter(lambda p: p.requires_grad, segmentor.parameters())
    opt = optim.SGD(learned_params,
                    lr=args.lr, momentum=0.99, weight_decay=0.0005)

    iter_order = int(np.log10(args.max_iter) + 1 )  # for pretty printing

    epoch = 0
    iteration = 0
    grad_counter = 0
    loss_meter = LossMeter()
    #while iteration < args.max_iter:
    while epoch < args.max_epoch:
        epoch += 1
        for im, target, _ in loader:
            im = im.to(device)
            target = target.to(device, non_blocking=True)

            seg = segmentor(im)
            loss = loss_fn(seg, target)
            loss.backward()
            loss_meter.update(loss.item())

            grad_counter += 1
            if grad_counter == args.iter_size:
                opt.step()
                opt.zero_grad()
                iteration += 1
                grad_counter = 0

                if iteration % loss_meter.average_over == 0:
                    log.info(f"epoch {epoch} iter {iteration:{iter_order}d} "
                            f"loss {loss_meter}")
                if (iteration % args.checkpoint_interval == 0
                    or iteration == args.max_iter):
                    log.info("checkpointing...")
                    torch.save(segmentor.state_dict(),
                            f'{output_dir}/checkpoint-'
                            f'iter{iteration:0{iter_order}d}.pth')
                    if do_eval:
                        log.info("evaluating...")
                        segmentor.eval()
                        eval_loss_meter, eval_seg_meter = evaluate(
                                segmentor, eval_dataset, eval_loader, device)
                        segmentor.train()
                        avg_loss = (eval_loss_meter.total_loss
                                    / len(eval_dataset))
                        log.info(f"loss {avg_loss:06.1f}")
                        log.info(f"metrics\n{eval_seg_meter}")
                        np.savez(f'{output_dir}/eval-'
                                 f'iter{iteration:0{iter_order}d}.npz',
                                 loss=avg_loss, hist=eval_seg_meter.hist)

            if iteration == args.max_iter:
                break


if __name__ == '__main__':
    parser = prepare_parser(train=True)
    args = parser.parse_args()
    main(args)
