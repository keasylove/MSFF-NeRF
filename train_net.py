from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)


    #add
    # maxPSNR = -1
    # tempMaxPsnr = -1

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        # add
        cfg.eval = False

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        # if (epoch + 1) % cfg.eval_ep == 0:
        #     trainer.val(epoch, val_loader, evaluator, recorder)

        #add
        # cfg.eval = True
        # maxPSNR = trainer.val(epoch, val_loader, evaluator, recorder, maxPSNR)
        #
        # if maxPSNR > tempMaxPsnr:
        #     tempMaxPsnr = maxPSNR
        #     if (epoch >= 10):
        #         save_model(network, optimizer, scheduler, recorder,
        #                    cfg.trained_model_dir, epoch)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    network = make_network(cfg)

    #add训练图
    # import wandb
    # wandb.init(
    #     # Set the project where this run will be logged
    #     project="msff-nerf-ZJUMOCAP",
    #     name=f"{cfg.exp_name}_F:{cfg.num_train_frame}",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": cfg.train.lr,
    #         "model": 'SMPL',
    #         "dataset": f"{cfg.train.dataset}-zju-mocap",
    #         "epochs": 400,
    #     })
    # wandb.login()

    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
