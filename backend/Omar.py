import torch
import torchio as tio
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fn
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms.functional import gaussian_blur

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (256, 256, 1)
OVERLAP = (64, 64, 0) 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225] 
MEAN = torch.tensor(mean).view(-1, 1, 1)
STD = torch.tensor(std).view(-1, 1, 1)

def load_model():
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, 21, 1)
    model.load_state_dict(torch.load("/data/models/best_model.pth", map_location=DEVICE)['model_state'])
    model = model.to(DEVICE).eval()
    return model

def Inference(model, img_path):
    # Preprocessing 
    img = Image.open(img_path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = (img - MEAN) / STD
    img = img.unsqueeze(0) # [1, 3, 1024, 2048]
    
    with torch.no_grad():       
        if img.ndim == 4: # [B, C, H, W]
            img = img.unsqueeze(-1) # [B, C, H, W, 1]

        for i in range(img.shape[0]):
            single_img = img[i] # [C, H, W, 1]
            
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=single_img)
            )
            
            sampler = tio.inference.GridSampler(
                subject,
                patch_size=PATCH_SIZE,
                patch_overlap=OVERLAP,
            )
            
            aggregator = tio.GridAggregator(sampler, overlap_mode="average")
            
            patch_loader = torch.utils.data.DataLoader(sampler, batch_size=4)
            
            for patch_batch in patch_loader:
                # TorchIO gives [B, C, H, W, 1]. Model wants [B, C, H, W]
                patch_images = patch_batch["image"][tio.DATA].squeeze(-1).to(next(model.parameters()).device, non_blocking=True)

                # Forward pass -> [B, C, H, W]
                logits = model(patch_images)['out']
                
                # Aggregator expects [B, C, H, W, D]. Add D=1 back.
                logits_5d = logits.unsqueeze(-1) 

                
                aggregator.add_batch(
                    logits_5d.cpu(),
                    patch_batch[tio.LOCATION]
                )
        
            output = aggregator.get_output_tensor() # [Classes, H, W, 1]
            


            full_logits = output.squeeze(-1) 
            pred = torch.argmax(full_logits, dim=0).cpu().numpy()


            pred_vis = np.where(pred == 20, 255, pred)
            # save colorful image
            # save black and white
            # [1,1024,2048]
            fig, ax = plt.subplots(figsize=(6, 6))

            im2 = ax.imshow(pred_vis, cmap="tab20", vmin=0, vmax=21)
            cbar = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title("Prediction")
            ax.axis("off")
            
            plt.tight_layout()
            colorful_path = img_path.replace(".png","prediction_with_colorbar.png")
            mask_path = img_path.replace(".png","prediction.png")
            plt.savefig(colorful_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            mask_bw = np.where(pred_vis > 19, 255, 0).astype(np.uint8)
            Image.fromarray(mask_bw).save(mask_path)

    return {"colorful": colorful_path, "mask" : mask_path}
