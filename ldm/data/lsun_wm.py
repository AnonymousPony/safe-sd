import os
import numpy as np
import PIL
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 wm_dir=None,
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        self.wm_len = 0
        if wm_dir is not None: 
            self.wm_imgs_path = self.get_wm_imgs_path(wm_dir)
            self.wm_len = len(self.wm_imgs_path)
        
    def get_wm_imgs_path(self, wm_dir):
        imgs_path = []
        for cur_img in os.listdir(wm_dir):
            wm_path = os.path.join(wm_dir, cur_img)
            imgs_path.append(wm_path)
            # image = Image.open(wm_path)
            # if not image.mode == "RGB":
            #     image = image.convert("RGB")
                
            # if self.size is not None:
            #     image = image.resize((self.size, self.size), resample=self.interpolation)
            # image = self.flip(image)
            # image = np.array(image).astype(np.uint8)
            # imgs.append((image / 127.5 - 1.0).astype(np.float32))
        return imgs_path

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        wm_idx = random.randint(0, self.wm_len-1)
        wm_image = Image.open(self.wm_imgs_path[wm_idx])
        if not wm_image.mode == "RGB":
            wm_image = wm_image.convert("RGB")
        
        wm_img = np.array(wm_image).astype(np.uint8)
        crop = min(wm_img.shape[0], wm_img.shape[1])
        wm_h, wm_w, = wm_img.shape[0], wm_img.shape[1]
        wm_img = wm_img[(wm_h - crop) // 2:(wm_h + crop) // 2, (wm_w - crop) // 2:(wm_w + crop) // 2]

        wm_image = Image.fromarray(wm_img)
        if self.size is not None:
            wm_image = wm_image.resize((self.size, self.size), resample=self.interpolation)

        wm_image = self.flip(wm_image)
        wm_image = np.array(wm_image).astype(np.uint8)        
        example["wm_img"] = (wm_image / 127.5 - 1.0).astype(np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/church_train", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/church_val",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
