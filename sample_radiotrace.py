import numpy as np
import yaml
import argparse
import math
import torch
from collections import defaultdict
import torch.nn as nn
from lib import loaders
from tqdm.auto import tqdm
from ema_pytorch import EMA
# from numpy import *
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import cluster as cl
import argparse


def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    criterion = nn.MSELoss()

    loss1 = criterion(pred1, target)/criterion(target, 0*target)
    loss2 = torch.sqrt(criterion(pred2, target))

    ssim1 = ssim(pred1, target)
    #ssim2 = ssim(pred2, target)

    psnr1 = psnr(pred1,target,data_range=1)

    metrics['nmse'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['rmse'] += loss2.data.cpu().numpy() * target.size(0)
    metrics['ssim'] += ssim1.data.cpu().numpy() * target.size(0)
    metrics['psnr'] += psnr1.data.cpu().numpy() * target.size(0)

    return [loss1,loss2]

def print_metrics_test(metrics, epoch_samples, error, results_folder):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format("Test"+" "+error, ", ".join(outputs)))
    with open(results_folder / "0_results.txt", "a") as f:
        f.write("{}: {}".format("Test"+" "+error, ", ".join(outputs)))
        f.write("\n")
def parse_args():
    parser = argparse.ArgumentParser(description="configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, default='configs/radio_sample_DDPM_by_y_MultiTx_BScorr_seenmean_SGDM_vqsg_heuristic.yaml')
    # data
    parser.add_argument('--data_dir', default='./RadioMapSeer/')

    # model
    parser.add_argument('--model_path', default='./model-250.pt')

    # scen
    parser.add_argument('--type', type=str, default='restrict_wo_BS', choices=['random', 'restrict_wo_BS'])
    parser.add_argument('--rate', type=float, default=0.01, help='Sampling rate')
    parser.add_argument('--num_forbidden', type=int, default=2, choices='Number of forbidden area')
    parser.add_argument('--rad', type=float, default=50, choices='radius of forbidden area')

    # RadioTrace
    parser.add_argument('--lr', type=float, default=333, help='Learning rate of Tx update')
    parser.add_argument('--momentum', type=float, default=0.4, help='momentum')
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--cluster_init', action='store_true')
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

# Colors for all 20 parts
part_colors = [[0, 0, 0], [255, 85, 0],  [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(51)
    np.random.seed(51)
    model_cfg = cfg.model
    # first_stage_cfg = model_cfg.first_stage
    # first_stage_model = AutoencoderKL(
    #     ddconfig=first_stage_cfg.ddconfig,
    #     lossconfig=first_stage_cfg.lossconfig,
    #     embed_dim=first_stage_cfg.embed_dim,
    #     ckpt_path=first_stage_cfg.ckpt_path,
    # )

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion, DDPM
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = DDPM(
        model=unet,
        # auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        # scale_factor=model_cfg.scale_factor,
        # scale_by_std=model_cfg.scale_by_std,
        # scale_by_softsign=model_cfg.scale_by_softsign,
        # default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))

    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDatasetTest(
            data_root=data_cfg.img_folder,
            image_size=model_cfg.image_size,
        )
        # dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    elif data_cfg['name'] == 'radio':
        dataset = loaders.RadioUNet_c_multiTx(phase="test", numTx=5, dir_dataset=args.data_dir)
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))


    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,cfg=cfg,args=args
    )
    sampler.sample()
    if data_cfg.name == 'cityscapes' or data_cfg.name == 'sr' or data_cfg.name == 'edge':
        exit()
    else:
        # assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
        # sampler.cal_fid(target_path=sampler_cfg.target_path)
        pass
    


def nmse(res, target):
    criterion = nn.MSELoss()
    return criterion(res, target) / criterion(target, 0 * target)


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
            args=None
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.image_size
        self.cfg = cfg

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        # self.results_folder_cond = Path(results_folder+'_cond')
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
            # self.results_folder_cond.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(args.model_path, map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']
        self.args = args

    def get_mask(self, type_mask='measure', **kwargs):
        def padding_mask():
            # padding mask
            mask = torch.zeros(1, 1, 256, 256)
            mask[:, :, 64:64+128, 64:64+128] = torch.ones(1, 1, 128, 128)
            return mask

        # random mask
        def random_mask(seen_ratio=0.01):
            mask = (torch.rand(1, 1, 256, 256)>1-seen_ratio).float()
            return mask

        # measure mask
        def measure_mask(num=4, r=50):
            loc = torch.randint(0, 255, (num, 2)).float()
            mask = torch.zeros(1, 1, 256, 256).float()
            for i in range(256):
                for j in range(256):
                    min_dis = torch.min((loc - torch.tensor([i, j]).expand(num, 2)).pow(2).sum(-1).sqrt())
                    if min_dis < r:
                        mask[0, 0, i, j] = 1
            return mask
        
        # restrict mask
        def restrict_mask(num=4, r=50, seen_ratio=0.01):
            loc = torch.randint(0, 255, (num, 2)).float()
            rd = torch.rand(1, 1, 256, 256)
            mask = torch.zeros(1, 1, 256, 256).float()
            for i in range(256):
                for j in range(256):
                    min_dis = torch.min((loc - torch.tensor([i, j]).expand(num, 2)).pow(2).sum(-1).sqrt())
                    if min_dis >= r:
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1-seen_ratio
            return mask

        def restrict_mask_without_BS(x, y, num=4, r=50, seen_ratio=0.01):
            loc = torch.randint(0, 255, (num, 2)).float()
            while 1:
                min_dis = torch.cdist(loc, torch.stack([x, y], dim=1).float().to('cpu')).min()
                # min_dis = torch.min((loc - torch.tensor([x, y]).expand(num, 2)).pow(2).sum(-1).sqrt())
                if min_dis < r:
                    break
                loc = torch.randint(0, 255, (num, 2)).float()
            rd = torch.rand(1, 1, 256, 256)
            mask = torch.zeros(1, 1, 256, 256).float()
            for i in range(256):
                for j in range(256):
                    min_dis = torch.min((loc - torch.tensor([i, j]).expand(num, 2)).pow(2).sum(-1).sqrt())
                    if min_dis >= r:
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1-seen_ratio
            return mask
        
        def nonuniform_mask(seen_ratio_1=0.01, seen_ratio_2=0.05):
            horize = 0 if torch.rand(()) < 0.5 else 1
            mask = torch.zeros(1, 1, 256, 256).float()
            if torch.rand(()) < 0.5:
                seen_ratio_1, seen_ratio_2 = seen_ratio_2, seen_ratio_1
            rd = torch.rand(1, 1, 256, 256)
            if horize:
                for i in range(256):
                    for j in range(256//2):
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1 - seen_ratio_1
                for i in range(256):
                    for j in range(256//2, 256):
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1 - seen_ratio_2
            else:
                for i in range(256//2):
                    for j in range(256):
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1 - seen_ratio_1
                for i in range(256//2, 256):
                    for j in range(256):
                        mask[0, 0, i, j] = rd[0, 0, i, j] > 1 - seen_ratio_2
            return mask
        
        if type_mask == 'padding':
            return padding_mask()
        elif type_mask == 'random':
            seen_ratio = 0.01 if 'seen_ratio' not in kwargs else kwargs['seen_ratio']
            return random_mask(seen_ratio=seen_ratio)
        elif type_mask == 'measure':
            num = 4 if 'num' not in kwargs else kwargs['num']
            r = 50 if 'r' not in kwargs else kwargs['r']
            return measure_mask(num=num, r=r)
        elif type_mask == 'restrict':
            num = 4 if 'num' not in kwargs else kwargs['num']
            r = 50 if 'r' not in kwargs else kwargs['r']
            seen_ratio = 0.01 if 'seen_ratio' not in kwargs else kwargs['seen_ratio']
            return restrict_mask(num=num, r=r, seen_ratio=seen_ratio)
        elif type_mask == 'restrict_wo_BS':
            assert 'x' in kwargs and 'y' in kwargs
            num = 4 if 'num' not in kwargs else kwargs['num']
            r = 50 if 'r' not in kwargs else kwargs['r']
            seen_ratio = 0.01 if 'seen_ratio' not in kwargs else kwargs['seen_ratio']
            return restrict_mask_without_BS(x=kwargs['x'], y=kwargs['y'], num=num, r=r, seen_ratio=seen_ratio)
        elif type_mask == 'nonuniform':
            seen_ratio_1 = 0.01 if 'seen_ratio_1' not in kwargs else kwargs['seen_ratio_1']
            seen_ratio_2 = 0.05 if 'seen_ratio_2' not in kwargs else kwargs['seen_ratio_2']
            return nonuniform_mask(seen_ratio_1=seen_ratio_1, seen_ratio_2=seen_ratio_2)
    
    def get_bs_from_xy(self, xy, size=256):
        # xy: [1, numTx, 2]
        # out: [1, 1, 256, 256]
        sigma = 1
        device = xy.device

        batch_size = xy.shape[0]
        numTx = xy.shape[1]
        R = torch.arange(size).unsqueeze(-1).expand(size, size).expand(batch_size, 1, size, size).to(device)
        C = torch.arange(size).unsqueeze(0).expand(size, size).expand(batch_size, 1, size, size).to(device)

        

        M = torch.zeros(batch_size, 1, size, size, dtype=torch.float32, device=device)
        for i in range(numTx):
            pos_x = xy[:, i, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size)
            pos_y = xy[:, i, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size)
            M = M + torch.exp(- ((R - pos_x)**2 + (C - pos_y)**2) / sigma)
        return M



    def sample(self):
        seen_ratio = self.args.rate
        type_mask = self.args.type # random, restrict_wo_BS, nonuniform
        type_cluster = 'v1' if self.args.cluster_init else 'none' # none, v1, v2


        metrics = defaultdict(float)
        accelerator = self.accelerator
        device = accelerator.device
        epoch_samples = 0
        batch_num = self.batch_num
        # with torch.no_grad():
        self.model.eval()
        psnr = 0.
        num = 0
        nmse_ = []
        rmse_ = []
        ssim_ = []
        psnr_ = []
        for idx, batch in tqdm(enumerate(self.dl)):

            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key].to(device)


            # image = batch["image"]
            # image = unnormalize_to_zero_to_one(image)
            cond = batch['cond']

            ######################
            #
            BS = cond[:, 1:2]
            Tx_pos = torch.where(BS[0, 0] == 1)
            numTx = len(Tx_pos[0])
            # Tx_pos = torch.argmax(torch.reshape(BS, (BS.shape[0], -1)), dim=-1)
            png_name = ''
            for i in range(Tx_pos[0].shape[0]):
                png_name += f"{Tx_pos[0][i]:.2f}_{Tx_pos[1][i]:.2f}-"
            tv.utils.save_image(BS[0]*0.5+0.5, str(self.results_folder / batch["img_name"][0])[:-4] + f"_Tx_{png_name}.png")
            cond[:, 1] = -1.0 * torch.ones_like(cond[:, 1], dtype=cond.dtype)
            # cond[:, 1, 255, 255] = 1
            
            ########################
            # torch.optim.SGD
            GT = batch['image']
            print(GT.size())

            if type_mask == 'random':
                mask_y = self.get_mask(type_mask='random', seen_ratio=0.01).to(GT.device)
            elif type_mask == 'restrict_wo_BS':
                mask_y = self.get_mask(type_mask='restrict_wo_BS', x=Tx_pos[0], y=Tx_pos[1], num=self.args.num_forbidden, seen_ratio=seen_ratio, r=self.args.rad).to(GT.device)
            elif type_mask == 'nonuniform':
                mask_y = self.get_mask(type_mask='nonuniform', seen_ratio_2=0.05).to(GT.device)
            ############################################### add Noise
            y = mask_y.to(GT.device) * GT
            ###############################################
            # print(batch["raw_size"])
            # raw_w = batch["raw_size"][0].item()      # default batch size = 1
            # raw_h = batch["raw_size"][1].item()
            img_name = batch["img_name"][0]

            yy = y[0].min() * (torch.ones(1, 1, 256, 256, dtype=y.dtype).to(GT.device) - mask_y[0]) + y[0]
            tv.utils.save_image(yy*0.5+0.5, str(self.results_folder / batch["img_name"][0])[:-4] + "_y.png")
            if type_cluster == 'v1':
                pos = cl.kmeans(image=y[0], cond=cond[0, 0], mask=mask_y[0, 0], num_Tx=numTx).unsqueeze(0).float()
            elif type_cluster == 'none':
                pos = cl.random_center(image=y[0], cond=cond[0, 0], mask=mask_y[0, 0], num_Tx=numTx).unsqueeze(0).float()
            # pos = torch.argmax(torch.reshape(yy, (y.shape[0], -1)), dim=-1)
            # pos = torch.stack([pos // y.shape[-1], pos % y.shape[-1]], dim=1).float()
            pos.requires_grad = True
            a = torch.zeros_like(y)
            a[0, 0, torch.round(pos[0, :, 0]).long(), torch.round(pos[0, :, 1]).long()] = 1
            tv.utils.save_image(a[0], str(self.results_folder / batch["img_name"][0])[:-4] + "_kmeans.png")
            BSS = self.get_bs_from_xy(pos)
            cond[:, 1:2] = BSS / 0.5 - 1

            mask = batch['ori_mask'] if 'ori_mask' in batch else None
            bs = cond.shape[0]
            if self.cfg.sampler.sample_type == 'whole':
                batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
            elif self.cfg.sampler.sample_type == 'slide':
                batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [256, 256]), stride=self.cfg.sampler.stride, mask=mask, y=y, mask_y=mask_y, path=str(self.results_folder / batch["img_name"][0])[:-4], bs=pos.to(GT.device))
            else:
                raise NotImplementedError
            calc_loss_test(batch_pred.cpu(), batch_pred.cpu(), (GT * 0.5 + 0.5).cpu(), metrics,
                            'mse')
            epoch_samples += batch_pred.size(0)
            for j, (img, c) in enumerate(zip(batch_pred, cond)):
                file_name = self.results_folder / img_name
                tv.utils.save_image(img, str(file_name)[:-4] + "_recon.png")
                tv.utils.save_image(GT[j]*0.5+0.5, str(file_name)[:-4] + "_GT.png")
                yy = y[j].min() * (torch.ones(1, 1, 256, 256, dtype=y.dtype).to(GT.device) - mask_y[j]) + y[j]
                tv.utils.save_image(yy*0.5+0.5, str(file_name)[:-4] + "_y.png")

                nmse_.append(nmse(img.cpu(), (GT[0]*0.5+0.5).cpu()).detach().numpy())
                

                # if idx == 50:
                #      break
        print_metrics_test(metrics, epoch_samples, 'mse', self.results_folder)
        accelerator.print('sampling complete')
        # print(mean(nmse_))

    # ----------------------------------waiting revision------------------------------------
    def slide_sample(self, inputs, crop_size, stride, mask=None, y=None, mask_y=None, path=None, bs=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    e1 = e2 = None
                    aux_out = None
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample_given_y_MultiTx_BScorr_seenmean_SGDM_vqsg_heuristic(
                            batch_size=1, 
                            cond=crop_img, 
                            mask=mask, 
                            y=y, 
                            mask_y=mask_y, 
                            path=path, 
                            bs=bs,
                            args = self.args
                        )
                    e1 = e2 = None
                    aux_out = None
                else:
                    raise NotImplementedError
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                if aux_out is not None:
                    aux_out1 += F.pad(aux_out,
                                   (int(x1), int(aux_out1.shape[3] - x2), int(y1),
                                    int(aux_out1.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        # torch.save(count_mat, '/home/yyf/Workspace/edge_detection/codes/Mask-Conditioned-Latent-Space-Diffusion/checkpoints/count.pt')
        seg_logits = preds / count_mat
        aux_out1 = aux_out1 / count_mat
        # aux_out2 = aux_out2 / count_mat
        if aux_out is not None:
            return seg_logits, aux_out1
        return seg_logits

    def whole_sample(self, inputs, raw_size, mask=None):

        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits


    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def rk45_sample(self, batch_size):
        with torch.no_grad():
            # Initial sample
            # z = torch.randn(batch_size, 3, *(self.image_size))
            shape = (batch_size, 3, *(self.image_size))
            ode_sampler = get_ode_sampler(method='RK45')
            x, nfe = ode_sampler(model=self.model, shape=shape)
            x = unnormalize_to_zero_to_one(x)
            x.clamp_(0., 1.)
            return x, nfe

def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t, model_type='const'):
    """Get the drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # rsde = sde.reverse(score_fn, probability_flow=True)
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # vec_t = torch.ones(shape[0], device=x.device) * t
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      # if denoise:
      #   x = denoise_update_fn(model, x)

      # x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass