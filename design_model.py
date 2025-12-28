import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Custom U-Net architecture for crack segmentation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # Separable Conv 1
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Separable Conv 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

# Dataset class for loading images and masks
class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, limit=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        if limit:
            self.images = self.images[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', image.size, 0)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask) / 255.0  # Normalize to 0-1

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

def calculate_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    intersection = (preds * targets).sum(dim=(2,3))
    dice = (2 * intersection / (preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + 1e-7)).mean().item()
    return dice

# Training function
def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        total_dice = 0
        count = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation', leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice = calculate_metrics(torch.sigmoid(outputs), masks)
                total_dice += dice
                count += 1
        avg_dice = total_dice / count if count > 0 else 0
        # Convert to percentage
        avg_dice_pct = avg_dice * 100
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Dice: {avg_dice_pct:.2f}")
        # Save metrics for frontend
        metrics_dir = 'training_runs/evaluation'
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
                    f"Val Loss: {val_loss/len(val_loader):.4f} | "
                    f"Dice: {avg_dice_pct:.2f}\n")

# Main training script
if __name__ == '__main__':
    # Data directories
    train_img_dir = 'data/train/images'
    train_mask_dir = 'data/train/masks'
    val_img_dir = 'data/valid/images'
    val_mask_dir = 'data/valid/masks'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Datasets and loaders
    train_dataset = CrackDataset(train_img_dir, train_mask_dir, transform)
    val_dataset = CrackDataset(val_img_dir, val_mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)

    # Train
    train_model(model, train_loader, val_loader, epochs=30, device=device)

    # Save model
    torch.save(model.state_dict(), 'model.pth')