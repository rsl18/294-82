import os
import glob
import setproctitle
import matplotlib 
import scipy.misc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from segmentron.data import DATASETS, PREPROCS, prepare_data, prepare_loader
from segmentron.model import ARCHS, BACKBONES, HEADS, prepare_model
from segmentron.metrics import LossMeter, SegMeter
from segmentron.options import prepare_parser
from segmentron.util import setup_logging


def evaluate(segmentor, dataset, loader, device):
    loss_fn = nn.CrossEntropyLoss(reduction='sum',
            ignore_index=dataset.ignore_index)

    loss_meter, seg_meter = LossMeter(), SegMeter(dataset.num_classes)
    count = 0 
    for im, target, _ in loader:
        
        # save JPEG image
        img = im.cpu().numpy()[0]
        img = np.transpose(img, (1,2,0))
        scipy.misc.imsave('./og_imgs/'+str(count)+'.jpg', img)
        
        im = im.to(device)
        target = target.to(device, non_blocking=True)
        seg = segmentor(im)
        _, new_seg = torch.max(seg, 1)
        
        # save segmenation image
        new_seg = new_seg.cpu().numpy()[0]
        scipy.misc.toimage(new_seg, cmin=0.0, cmax=255.0).save('new_seg_imgs/'+str(count)+'.png')
        
        loss = loss_fn(seg, target)
        loss_meter.update(loss.item())
        seg_meter.update(seg, target)

        count += 1
    return loss_meter, seg_meter


def main(args):
    if os.path.isfile(args.output):
        # evaluate a single checkpoint
        evals = [args.output]
    elif os.path.isdir(args.output):
        # evaluate all checkpoints in a dir
        evals = sorted(glob.glob(os.path.join(args.output, '*.pth')))
    else:
        raise Exception(f"{arg.output} does not exist.")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_grad_enabled(False)  # inference only for memory efficiency

    dataset = prepare_data(args.eval_dataset, args.eval_split, args.preproc)
    loader = prepare_loader(dataset, evaluation=True)

    segmentor = prepare_model(args.arch, args.backbone, args.head,
                              dataset.num_classes, evals[0]).to(device)
    segmentor.eval()

    for weights in evals:
        logfile = f'{weights[:-4]}-{args.eval_output}.log'
        print(logfile)
        resultsfile = f'{weights[:-4]}-{args.eval_output}.npz'
        if os.path.isfile(resultsfile):
            print(f"skipping existing {resultsfile}")
            continue
        setproctitle.setproctitle(f"segeval-{weights}")
        log = setup_logging(logfile)
        log.info(f"args: {vars(args)}")
        log.info(f"weights: {weights}")

        segmentor.load_state_dict(torch.load(weights))
        loss_meter, seg_meter = evaluate(segmentor, dataset, loader, device)

        avg_loss = loss_meter.total_loss / len(dataset)
        log.info(f"loss {avg_loss:06.1f}")
        log.info(f"metrics\n{seg_meter}")

        np.savez(resultsfile, loss=avg_loss, hist=seg_meter.hist.cpu())


if __name__ == '__main__':
    parser = prepare_parser(train=False)
    args = parser.parse_args()
    main(args)
