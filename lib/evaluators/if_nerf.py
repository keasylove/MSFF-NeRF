import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
from termcolor import colored


class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        orig_img_pred = img_pred.copy()
        orig_img_gt = img_gt.copy()

        if 'crop_bbox' in batch:
            img_pred = fill_image(img_pred, batch)
            img_gt = fill_image(img_gt, batch)

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        # cv2.imwrite(
        #     '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
        #                                            view_index),
        #     (img_pred[..., [2, 1, 0]] * 255))
        # cv2.imwrite(
        #     '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
        #                                               view_index),
        #     (img_gt[..., [2, 1, 0]] * 255))

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = orig_img_pred[y:y + h, x:x + w]
        img_gt = orig_img_gt[y:y + h, x:x + w]
        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)

        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        return ssim

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        if rgb_gt.sum() == 0:
            return

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(result_path, metrics)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        self.mse = []
        self.psnr = []
        self.ssim = []

def fill_image(img, batch):
    orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
    full_img = np.zeros((orig_H, orig_W, 3))
    bbox = batch['crop_bbox'][0].detach().cpu().numpy()
    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    full_img[bbox[0, 1]:bbox[1, 1],
             bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
    return full_img
