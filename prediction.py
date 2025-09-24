import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import math
import models as Models
import csv
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io as sio
import re


class MeanAngularError(nn.Module):
    def __init__(self):
        super(MeanAngularError, self).__init__()

    @torch.no_grad()
    def forward(self, output, target, mask):
        dot_product = (output * target).sum(1)
        output_norm = torch.norm(output, 2, 1)
        target_norm = torch.norm(target, 2, 1)
        dot_product = (dot_product / (output_norm * target_norm + 1e-8)).clamp(-1, 1)
        angular_map = torch.acos(dot_product) * (180.0 / math.pi)
        angular_map = angular_map * mask[:, 0]
        return angular_map.sum() / mask.sum().item()

class CustomDataset(Dataset):
    def __init__(self, root_dir, file_list, train=False, augment=False):
        self.root_dir = root_dir
        self.file_paths = []
        self.augment = augment
        self.train = train
        self.crop_dir = os.path.join(os.path.dirname(self.root_dir), "crop_indices")

        with open(file_list, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.file_paths.append(os.path.join(root_dir, row[0]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data_path = self.file_paths[idx]
        data = torch.load(data_path)
        images = data['images']
        est = data['est']
        spec = data['spec']
        Iun = data['Iun']
        cos1 = data['cos1']
        cos2 = data['cos2']
        DoP = data['DoP']
        image_coordinate = data['image_coordinate']
        mask = data['mask']
        normals_gt = data['label']
        return images, est, spec, Iun, cos1, cos2, DoP, image_coordinate, mask, normals_gt

def load_data(batch_size, root_dir, test_list):
    test_dataset = CustomDataset(root_dir, test_list, train=False, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_loader

def anglar_within_angle(mae_map_valid, angle=11.25):
    return float((mae_map_valid < angle).sum()) / float(mae_map_valid.shape[0])

def visualize_output_only(sample, sample_name=None, figsize=(8,8), disp=False, dest_dir=None):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.tight_layout()
    if sample_name is not None:
        fig.suptitle(f'{sample_name}', y=1.01)

    gs = GridSpec(1, 1, figure=fig)
    normals_output_ax = fig.add_subplot(gs[0,0])

    out_img = sample['output'].cpu().detach().numpy()  # [3, H, W]
    mask = sample['mask'].cpu().detach().numpy()       # [1, H, W]

    mask_bool = (mask[0] == 1)
    out_img_masked = out_img.copy()
    for c in range(3):
        channel = out_img_masked[c]
        channel[~mask_bool] = np.nan  # mask外设为NaN
        out_img_masked[c] = channel
    out_img_masked = (out_img_masked + 1.0) / 2.0

    out_img_masked = np.transpose(out_img_masked, (1,2,0))
    normals_output_ax.imshow(out_img_masked)
    normals_output_ax.set_title('Predicted Normals')

    if dest_dir is not None:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_png = os.path.join(dest_dir, f"{sample_name.replace('/', '-')}.png")
        fig.savefig(dest_png, facecolor='w')
        dest_mat = os.path.join(dest_dir, f"{sample_name.replace('/', '-')}.mat")
        sio.savemat(dest_mat, {'normals': out_img_masked})
    plt.close()

def test_model(model, dataloader, device, img_save_dir='./revises'):
    model.eval()
    all_valid_pixels = []

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    with torch.no_grad():
        for i, (images,est,spec,Iun,cos1,cos2,DoP,image_coordinate,mask,normals_gt) in enumerate(dataloader):
            images = images.to(device)
            est = est.to(device)
            spec = spec.to(device)
            Iun = Iun.to(device)
            cos1 = cos1.to(device)
            cos2 = cos2.to(device)
            DoP = DoP.to(device)
            image_coordinate = image_coordinate.to(device)
            mask = mask.to(device)
            normals_gt = normals_gt.to(device)
            print(images.shape)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            output = model(images, est, spec, Iun, cos1, cos2, DoP, image_coordinate)
            end_time.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            print(f"Inference time: {inference_time:.2f} ms")
            output = F.normalize(output, p=2, dim=1)

            sample = {
                'output': output[0],
                'mask': mask[0]
            }
            visualize_output_only(sample, disp=False, dest_dir=img_save_dir)

def calculate_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    root_dir = "./data/SurfaceNormals/objectsall1024"
    test_list = os.path.join(root_dir, "test_list.csv")
    models_path = './ckpt_epochs/model_epoch_final.pth'
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_save_dir = './normal'
    test_loader = load_data(batch_size, root_dir, test_list)

    residual_num = 8
    model = Models.get_model(residual_num)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(models_path, map_location=device))
    model = model.to(device)
    test_model(model, test_loader, device, img_save_dir=img_save_dir)
