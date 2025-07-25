import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.to_cuda(b) for b in batch]
            return batch

        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)

        return batch

    def add_iter_step(self, batch, iter_step):
        if isinstance(batch, tuple) or isinstance(batch, list):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step)

        if isinstance(batch, dict):
            batch['iter_step'] = iter_step

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)
            self.add_iter_step(batch, epoch * max_iter + iteration)
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            if cfg.training_mode == 'default':
                optimizer.zero_grad()
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                optimizer.step()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                exp_name = 'exp: {}'.format(cfg.exp_name)
                training_state = '  '.join([exp_name, 'eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None, maxPSNR=-1):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output = self.network(batch, True)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

        #     loss_stats = self.reduce_loss_stats(loss_stats)
        #     for k, v in loss_stats.items():
        #         val_loss_stats.setdefault(k, 0)
        #         val_loss_stats[k] += v
        #
        # loss_state = []
        # for k in val_loss_stats.keys():
        #     val_loss_stats[k] /= data_size
        #     loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        # print(loss_state)

        # if evaluator is not None:
        #     result = evaluator.summarize()
        #     val_loss_stats.update(result)
        #
        # if recorder:
        #     recorder.record('val', epoch, val_loss_stats, image_stats)

        #add
        import numpy as np
        import wandb
        PSNR = np.mean(evaluator.psnr)
        MSE = np.mean(evaluator.mse)
        SSIM = np.mean(evaluator.ssim)

        if PSNR > maxPSNR:
            maxPSNR = PSNR
            # wandb.log({'current_Max_PSNR': maxPSNR, 'current_MAx_PSNR_epoch': epoch, 'epoch':epoch}) # 训练图

        # if evaluator is not None: # 训练图
        #     wandb.log({'Test-PSNR': PSNR, 'Test-MSE': MSE, 'Test-SSIM': SSIM,  'epoch':epoch}) # 训练图

        evaluator.psnr = []
        evaluator.mse = []
        evaluator.ssim = []

        return maxPSNR
