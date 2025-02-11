import os
import numpy as np
import torch
import cv2
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Trainer:
    def __init__(self, data_dir, checkpoint_path, model_cfg, device="cuda"):
        self.data_dir = data_dir
        self.device = device

        # Load model
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

        # Enable training for components
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)
        # Uncomment below if you want to train the image encoder (requires removing "no_grad" in SAM2 code)
        # self.predictor.model.image_encoder.train(True)

        # Optimizer and scaler for mixed precision
        self.optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
        self.scaler = torch.cuda.amp.GradScaler()

        # Load dataset
        self.data = self._load_data()

    def _load_data(self):
        """Load dataset file paths."""
        data = []
        for name in os.listdir(os.path.join(self.data_dir, "Simple/Train/Image/")):
            data.append({
                "image": os.path.join(self.data_dir, "Simple/Train/Image/", name),
                "annotation": os.path.join(self.data_dir, "Simple/Train/Instance/", name[:-4] + ".png")
            })
        return data

    def _read_batch(self):
        """Read a random image and its annotation from the dataset."""
        entry = self.data[np.random.randint(len(self.data))]
        image = cv2.imread(entry["image"])[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(entry["annotation"])

        # Resize image
        scale = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * scale), int(ann_map.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

        # Merge annotations
        mat_map = ann_map[:, :, 0]
        ves_map = ann_map[:, :, 2]
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

        # Create binary masks and points
        indices = np.unique(mat_map)[1:]
        masks, points = [], []
        for idx in indices:
            mask = (mat_map == idx).astype(np.uint8)
            masks.append(mask)
            coords = np.argwhere(mask > 0)
            point = np.array(coords[np.random.randint(len(coords))])
            points.append([[point[1], point[0]]])

        return image, np.array(masks), np.array(points), np.ones([len(masks), 1])

    def _compute_loss(self, gt_mask, prd_masks, prd_scores):
        """Compute segmentation and score loss."""
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

        return seg_loss + score_loss * 0.05, iou

    def run(self, num_iterations=100000, save_interval=1000):
        """Run the training process."""
        mean_iou = 0
        for itr in range(num_iterations):
            with torch.cuda.amp.autocast():
                # Load batch data
                image, masks, input_points, input_labels = self._read_batch()
                if masks.shape[0] == 0:
                    continue

                self.predictor.set_image(image)

                # Prompt encoding
                mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                    input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
                )
                sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )

                # Mask decoder
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features
                )
                prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

                # Compute losses
                gt_mask = torch.tensor(masks.astype(np.float32)).cuda()
                loss, iou = self._compute_loss(gt_mask, prd_masks, prd_scores)

            # Backpropagation
            self.predictor.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Save model periodically
            if itr % save_interval == 0:
                torch.save(self.predictor.model.state_dict(), "model.torch")
                print("Model saved at iteration", itr)

            # Display results
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Step: {itr}, Accuracy (IOU): {mean_iou:.4f}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM2 Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to the model configuration file")
    args = parser.parse_args()

    trainer = SAM2Trainer(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        model_cfg=args.model_cfg
    )
    trainer.run()

# python train.py --data_dir /mnt/data/segmentation/LabPicsV1/ --checkpoint_path ../backup/checkpoints/sam2_hiera_small.pt --model_cfg sam2_hiera_s.yaml