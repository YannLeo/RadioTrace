import numpy as np
np.random.seed(42)
import torch

def dis2sig(x, a=-0.22221996, b=0.58683092, c=0.26315023, d=0.89743365):
    return a * torch.log(b * x + c) + d

def sig2dis(x, a=-0.22221996, b=0.58683092, c=0.26315023, d=0.89743365):
    return (torch.exp((x - d) / a) - c) / b

def random_center(image: torch.tensor, cond: torch.tensor, mask: torch.tensor, num_Tx):
    if image.dim() == 3:
        image = image[0]
    w, h = image.shape
    image_clone = image.clone()
    centroids = torch.zeros(num_Tx, 2)

    values = torch.arange(w)
    pairs = torch.cartesian_prod(values, values)

    # prepare the sample point
    pos = torch.where(mask > 0.5)
    sample = torch.stack([pos[0], pos[1], image[pos]], dim=1)
    pos = torch.where(sample[:, 2] > 0)
    sample = sample[pos[0], :]

    if len(sample) <= 0:
        return torch.from_numpy(np.random.randint(0, w, (num_Tx, 2))).to(sample.device)
    
    # init the centroids
    for i in range(num_Tx):
        x, y = torch.where(image_clone == image_clone.max())
        centroids[i, 0] = x[0] + 0.1 * np.random.randn()
        centroids[i, 1] = y[0] + 0.1 * np.random.randn()
        
        distance = (pairs - centroids[i]).pow(2).sum(-1).sqrt()
        idx = torch.where(distance <= 15)
        pos = pairs[idx]
        image_clone[pos[:, 0], pos[:, 1]] = -1

    centroids = centroids.to(sample.device)
    return centroids


def kmeans(image: torch.tensor, cond: torch.tensor, mask: torch.tensor, num_Tx, max_iter=100, tol=1e-2):
    if image.dim() == 3:
        image = image[0]
    w, h = image.shape
    image_clone = image.clone()
    centroids = torch.zeros(num_Tx, 2)

    values = torch.arange(w)
    pairs = torch.cartesian_prod(values, values)

    # prepare the sample point
    pos = torch.where(mask > 0.5)
    sample = torch.stack([pos[0], pos[1], image[pos]], dim=1)
    pos = torch.where(sample[:, 2] > 0)
    sample = sample[pos[0], :]

    if len(sample) <= 0:
        return torch.from_numpy(np.random.randint(0, w, (num_Tx, 2))).to(sample.device)
    
    # init the centroids
    for i in range(num_Tx):
        x, y = torch.where(image_clone == image_clone.max())
        centroids[i, 0] = x[0] + 0.1 * np.random.randn()
        centroids[i, 1] = y[0] + 0.1 * np.random.randn()
        
        distance = (pairs - centroids[i]).pow(2).sum(-1).sqrt()
        idx = torch.where(distance <= 50)
        pos = pairs[idx]
        image_clone[pos[:, 0], pos[:, 1]] = -1

    centroids = centroids.to(sample.device)
    base_centroids = centroids.clone()

    for _ in range(max_iter):
        distance = torch.cdist(sample[:, :2], centroids.to(sample.device))
        cluster_idx = torch.argmin(distance, dim=1)

        new_centroids = centroids.clone()

        for i in range(num_Tx):
            idx = torch.where(cluster_idx == i)[0]
            if len(idx) <= 0:
                new_centroids[i] = base_centroids[i]
                continue
            value_idx = sample[idx, 2]
            pos_idx = sample[idx, :2]
            dis_eval = sig2dis(value_idx)
            direction = centroids[i] - pos_idx
            direction =  direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-4) * dis_eval.unsqueeze(-1)
            new_centroids[i] = (value_idx.unsqueeze(-1) * (sample[idx, :2] + direction)).sum(dim=0) / (value_idx.sum())
        
        max_update = (centroids - new_centroids).pow(2).sum(-1).sqrt().max()
        if max_update < tol:
            centroids = new_centroids
            break

        centroids = new_centroids
        # print(centroids)


    centroids = torch.clip(centroids, 0, w-1)
    if cond[torch.round(centroids[:, 0]).long(), torch.round(centroids[:, 1]).long()].max() > 0.5:
        for i in range(num_Tx):
            x, y = centroids[i]
            if cond[x.long(), y.long()] > 0.5:
                # find the nearest point where cond < 0.5
                empty_pos = torch.where(cond < 0.5)
                empty_pos = torch.stack(empty_pos, dim=1)
                distance = (empty_pos - centroids[i]).pow(2).sum(-1).sqrt()
                new = empty_pos[torch.argmin(distance)]
                centroids[i] = new

    return centroids

    # 检查是否为建筑物

if __name__ == '__main__':
    def get_mask(type_mask='measure', **kwargs):
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
                min_dis = torch.min((loc - torch.tensor([x, y]).expand(num, 2)).pow(2).sum(-1).sqrt())
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
        


    from lib.loaders import *
    import torchvision as tv
    dataset = RadioUNet_c_multiTx(phase='train')
    
    idx = 1000
    data = dataset[idx]
    BS = data['cond'][1]
    image = data['image']
    buildings = data['cond'][0]
    mask = get_mask('restrict', seen_ratio=0.01)[0, 0]
    y = mask * image
    Tx_pos = torch.where(BS > 0.5)
    # Tx_pos = torch.argmax(torch.reshape(BS, (1, -1,)), dim=-1)
    tv.utils.save_image(image*0.5+0.5, str('temp/16_1BS/'  + str(idx) + f"_Tx.png"))
    tv.utils.save_image((mask * image + (1-mask) * -1 * torch.ones_like(image))*0.5+0.5, str('temp/16_1BS/'  + str(idx) + f"_Tx_mask.png"))
    print(Tx_pos)
    center = kmeans(y, buildings, mask, len(Tx_pos[0]))
    a = torch.zeros_like(image)
    a[0, torch.round(center[:, 0]).long(), torch.round(center[:, 1]).long()] = 1
    tv.utils.save_image(a, 'temp/16_1BS/km.png')
    pass
