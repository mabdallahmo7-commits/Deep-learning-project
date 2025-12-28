import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import glob

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

def calculate_dice_coefficient(pred_mask, gt_mask):
    """
    Calculates the Dice Coefficient between two binary masks.
    """
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    intersection = np.logical_and(pred_mask, gt_mask)
    
    if (np.sum(pred_mask) + np.sum(gt_mask)) == 0:
        return 1.0 # Both masks are empty
        
    dice_score = 2. * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask))
    return dice_score

def test_model():
    # This function calculates total performance and saves predicted masks.
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # Directories for training data
    train_image_dir = 'data/test/images'
    train_mask_dir = 'data/test/masks'
    
    train_image_paths = glob.glob(os.path.join(train_image_dir, '*.jpg'))

    output_dir = 'test_masks'
    os.makedirs(output_dir, exist_ok=True)

    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for img_path in train_image_paths:
            base_name_with_ext = os.path.basename(img_path)
            base_name = os.path.splitext(base_name_with_ext)[0]

            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # corresponding mask and label paths
            mask_path = os.path.join(train_mask_dir, base_name + '.png')

            if os.path.exists(mask_path):
                gt_mask_img = Image.open(mask_path).convert('L')
                gt_mask_tensor = mask_transform(gt_mask_img)
                gt_mask = np.array(gt_mask_tensor)

                output = model(input_tensor)
                output_mask_pred = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
                
                dice = calculate_dice_coefficient(output_mask_pred, gt_mask)
                
                print(f"Processing {base_name_with_ext}: Dice Score = {dice:.4f}")

                # Save predicted mask
                pred_img = Image.fromarray((output_mask_pred * 255).astype(np.uint8))
                pred_img.save(os.path.join(output_dir, base_name + '.png'))

                total_dice += dice
                count += 1
            else:
                print(f"Warning: Corresponding mask not found for {img_path}. Skipping.")
    
    if count > 0:
        print(f"\nTotal Performance (Average Dice Score): {total_dice / count:.4f}")
    else:
        print("\nNo images processed.")

if __name__ == '__main__':
    test_model()