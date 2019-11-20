# Future imports:
from __future__ import absolute_import, division, print_function

# Standard Library imports:
import os

# 3rd Party imports:
from progress.bar import Bar
import torch
import torch.utils.data

import _init_paths
from detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
from logger import Logger
from models.data_parallel import DataParallel
from models.model import create_model, load_model, save_model
from opts import opts
from trains.train_factory import train_factory
from utils.utils import AverageMeter

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  print("ARCH: ", opt.arch)
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    print("LOAD MODEL IN MAIN: ", opt.load_model)
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir, logger)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))

    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)

      with torch.no_grad():
        if uses_coco_eval(opt):
            do_eval(opt, epoch, model, Dataset, logger)
        else:
            log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary("val_{}".format(k), v, epoch)
                logger.write("{} {:8f} | ".format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(
                    os.path.join(opt.save_dir, "model_best.pth"), epoch, model
                )
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()


def uses_coco_eval(opt) -> bool:
    return opt.dataset in ["ucb_coco", "xview", "coco"]


def do_eval(opt, epoch, model, DatasetFactory, logger):
    # Based this code on test.py's non-prefetched code path:
    Detector = detector_factory[opt.task]
    dataset = DatasetFactory(opt, "val")
    detector = Detector(opt, model)
    best = float("-inf")
    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        if opt.task == 'ddd':
            ret = detector.run(img_path, img_info['calib'])
        else:
            ret = detector.run(img_path)

        results[img_id] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()

    # Capture metric of interest, e.g., for COCO eval, something like AP50:
    eval_stats = dataset.run_eval(results, opt.save_dir, logger)
    if uses_coco_eval(opt):
        metric = eval_stats[1]
        # Log results to log.txt and/or tensorboard:
        logger.scalar_summary("val_ap50", metric, epoch)
    else:
        # Pascal VOC:
        metric = eval_stats["Mean AP"]
        # Log results to log.txt and/or tensorboard:
        logger.scalar_summary("mean_AP", metric, epoch)

    # Best model checkpointing:
    if metric > best:
        best = metric
        save_model(
            os.path.join(opt.save_dir, "model_best.pth"), epoch, model
        )

    # Original attempt:
    # eval_stats = val_loader.dataset.run_eval(preds, opt.save_dir, logger)
    # ##AP50
    # metric = eval_stats[1]
    # logger.scalar_summary("val_ap50", metric, epoch)
    # logger.write("{} {:8f} | ".format("ap50", metric))
    # if metric > best:
    #     best = metric
    #     save_model(
    #         os.path.join(opt.save_dir, "model_best.pth"), epoch, model
    #     )


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)

