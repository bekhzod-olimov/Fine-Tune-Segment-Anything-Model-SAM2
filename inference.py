import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Inference:
    def __init__(self, image_path, mask_path, sam2_checkpoint, model_cfg, natijalar_uchun_papka, num_samples=30, device="cuda"):
        self.image_path = image_path
        self.mask_path = mask_path
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.num_samples = num_samples
        self.device = device
        self.natijalar_uchun_papka = natijalar_uchun_papka

        # # Modelni qurish va uni ishga tushirish
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.predictor.model.load_state_dict(torch.load("../backup/checkpoints/model.torch"))

    def read_image(self):  # Rasm va maskani o'qish va o'lchamini o'zgartirish
        # # RGB formatida rasmni o'qish
        img = cv2.imread(self.image_path)[..., ::-1]  
        # # Segmentatsiya qilinadigan hududni ko'rsatadigan maskani o'qish
        mask = cv2.imread(self.mask_path, 0)  

        # # Maksimal o'lchami 1024 ga teng bo'lgan rasmni o'lchash
        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def get_points(self, mask, num_points):  # Mask ichidagi tasodifiy nuqtalarni tanlash
        points = []
        for i in range(num_points):
            # # Maskdan ijobiy qiymatlar koordinatalarini topish
            coords = np.argwhere(mask > 0)
            yx = np.array(coords[np.random.randint(len(coords))])
            # # Nuqtalarni (x, y) formatida saqlash
            points.append([[yx[1], yx[0]]])  
        return np.array(points)

    def run(self):
        # # Rasm va maskani o'qish
        image, mask = self.read_image()

        # # Tasodifiy nuqtalarni tanlash
        input_points = self.get_points(mask, self.num_samples)

        # # bfloat16 formatida CUDA qurilmasidan foydalanish (xotirani tejash uchun)
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        # # Oldindan o'qitilgan model yordamida maskani bashorat qilish
        with torch.no_grad():
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )

        # # Yuqoridan pastgacha natijalarni saralash
        masks = masks[:, 0].astype(bool)
        sorted_masks = masks[np.argsort(scores[:, 0])][::-1].astype(bool)

        # # Bashorat qilingan barcha maskalarni birlashtirish
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

        for i in range(sorted_masks.shape[0]):
            mask = sorted_masks[i]
            # # Qaytariladigan hududlarni o'tkazib yuborish
            if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                continue
            mask[occupancy_mask] = 0
            seg_map[mask] = i + 1
            occupancy_mask[mask] = 1

        # # Segmentatsiya xaritasini rangli rasmga aylantirish
        rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for id_class in range(1, seg_map.max() + 1):
            rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

        # # Natijalarni saqlash
        
        os.makedirs(self.natijalar_uchun_papka, exist_ok=True)

        cv2.imwrite(f"{self.natijalar_uchun_papka}/bashorat_mask.png", rgb_image)
        cv2.imwrite(f"{self.natijalar_uchun_papka}/bashorat_aralash_mask.png", ((rgb_image / 2 + image / 2).astype(np.uint8)))

        print("Segmentatsiya bajarildi va natijalar saqlandi.")

# # Foydalanish uchun misol:
if __name__ == "__main__":
    image_path = r"sample_image.jpg"  # Rasm yo'li
    mask_path = r"sample_mask.png"   # Mask yo'li
    sam2_checkpoint = "../backup/checkpoints/sam2_hiera_small.pt"  # Model checkpoint
    model_cfg = "sam2_hiera_s.yaml"  # Model konfiguratsiyasi
    natijalar_uchun_papka = "natijalar"

    # # Inference obyektini yaratish va segmentatsiyani bajarish
    inference = Inference(image_path, mask_path, sam2_checkpoint, model_cfg, natijalar_uchun_papka)
    inference.run()
