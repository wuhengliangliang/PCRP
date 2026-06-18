# datasets/vg_dataset.py
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils import convert_examples_to_features, read_examples
from .transforms import PIL_TRANSFORMS
from transformers import BertTokenizer


SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
        'splits': ('train', 'val', 'test'),
        'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {'splits': ('train', 'val', 'test')},
    'pathology2': {'splits': ('train', 'val', 'testA', 'testB')},
}


class VGDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_root='data',
        dataset='referit',
        transforms=None,
        debug=False,
        test=False,
        split='train',
        max_query_len=128,
        bert_mode='bert-base-uncased',
        cache_images=False
    ):
        super().__init__()
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.test = test
        self.debug = debug
        self.query_len = max_query_len
        self.cache_images = cache_images

        transforms = transforms or []
        self.transforms = []
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)

        # dataset root
        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        elif self.dataset == 'pathology2':
            self.dataset_root = osp.join(self.data_root, 'pathology2')
            self.im_dir = osp.join(self.dataset_root, 'images')
        else:
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'COCO2014', 'train2014')

        dataset_split_root = osp.join(self.split_root, self.dataset)
        valid_splits = SUPPORTED_DATASETS[self.dataset]['splits']
        if split not in valid_splits:
            raise ValueError(f'Dataset {self.dataset} does not have split {split}')

        # load split info
        self.imgset_info = []
        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]

        for sp in splits:
            imgset_file = f'refpath_{sp}.pth'
            imgset_path = osp.join(dataset_split_root, imgset_file)
            print("imgset_path:", imgset_path)
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        if self.dataset == 'flickr':
            self.img_names, self.bboxs, self.phrases = zip(*self.imgset_info)
            self.reason = None
        else:
            # (img_name, ?, bbox, phrase, ?, reason)
            self.img_names, _, self.bboxs, self.phrases, _, self.reason = zip(*self.imgset_info)

        # convert bbox to xyxy in original image coords
        self.covert_bbox = []
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            for bbox in self.bboxs:
                bbox = np.array(bbox, dtype=np.float32)  # xywh
                bbox[2:] += bbox[:2]                     # -> xyxy
                self.covert_bbox.append(bbox)
        else:
            for bbox in self.bboxs:
                bbox = np.array(bbox, dtype=np.float32)
                self.covert_bbox.append(bbox)

        if cache_images:
            self.images_cached = [None] * len(self)
            self.read_image_orig_func = self.read_image_from_path_PIL
            self.read_image = self.read_image_from_cache
        else:
            self.read_image = self.read_image_from_path_PIL

    def __len__(self):
        return len(self.img_names)

    @staticmethod
    def _norm_slash(p: str) -> str:
        p = str(p).replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        return p

    def image_path(self, idx):
        """
        修复点：split 里 img_name 可能带了 'refpath_image/xxx.jpg' 前缀，
        但 self.im_dir 本身已经是 '/.../refpath_image'，直接 join 会变成 refpath_image/refpath_image/xxx。
        这里统一做：
        1) 规范化分隔符
        2) 剥掉重复前缀（按 im_dir basename / data_root basename / 常见 images/refpath_image）
        3) 多候选路径 fallback（保证最大兼容性）
        """
        raw = self.img_names[idx]
        name = self._norm_slash(raw)

        # 绝对路径直接返回
        if osp.isabs(name):
            p_abs = osp.normpath(name)
            if osp.exists(p_abs):
                return p_abs
            # 绝对路径但不存在，也允许继续走 fallback（有些人存了错误绝对路径）

        leaf_imdir = osp.basename(osp.normpath(self.im_dir))
        leaf_dataroot = osp.basename(osp.normpath(self.data_root))

        # 可能冗余的前缀（你现在就是 refpath_image 重复）
        prefixes = []
        if leaf_imdir:
            prefixes.append(leaf_imdir)
        if leaf_dataroot:
            prefixes.append(leaf_dataroot)
        prefixes += ["refpath_image", "images"]

        # 反复剥前缀：refpath_image/refpath_image/xxx -> xxx
        for pre in prefixes:
            if not pre:
                continue
            pre = self._norm_slash(pre).rstrip("/") + "/"
            while name.startswith(pre):
                name = name[len(pre):]

        base = osp.basename(name)  # 只取文件名（最后兜底常用）

        # 候选路径：从最可能到最不可能
        candidates = [
            osp.join(self.im_dir, name),
            osp.join(self.im_dir, base),
            osp.join(self.im_dir, "images", base),

            osp.join(self.data_root, name),
            osp.join(self.data_root, base),
            osp.join(self.data_root, "refpath_image", base),
            osp.join(self.data_root, "refpath_image", "images", base),

            osp.join(getattr(self, "dataset_root", ""), name),
            osp.join(getattr(self, "dataset_root", ""), "images", base),
        ]

        for p in candidates:
            if not p:
                continue
            p = osp.normpath(p)
            if osp.exists(p):
                return p

        raise FileNotFoundError(
            f"[VGDataset] image not found.\n"
            f"  img_names[{idx}] = {raw}\n"
            f"  cleaned_name      = {name}\n"
            f"  im_dir            = {self.im_dir}\n"
            f"  data_root         = {self.data_root}\n"
            f"  tried:\n    " + "\n    ".join([osp.normpath(x) for x in candidates if x])
        )

    def annotation_box(self, idx):
        return self.covert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        image_path = self.image_path(idx)
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        return self.images_cached[idx]

    def __getitem__(self, idx):
        # image
        image = self.read_image(idx)

        # bbox in original image coords (xyxy)
        bbox = torch.tensor(self.annotation_box(idx), dtype=torch.float32)

        phrase = str(self.phrase(idx)).lower()
        reasons = []
        if self.reason is not None:
            try:
                reasons = list(self.reason[idx])
            except Exception:
                reasons = []

        # pad reasons to 3 safely
        if len(reasons) == 0:
            reasons = [phrase]
        while len(reasons) < 3:
            reasons.append(reasons[-1])

        target = {}
        target['phrase'] = phrase
        target['bbox'] = bbox

        # ✅ ALWAYS keep orig_bbox for debug (train/val/test)
        target['orig_bbox'] = bbox.clone()

        # transforms (should fill size/ratio/dxdy/orig_size etc)
        for transform in self.transforms:
            image, target = transform(image, target)

        # BERT main phrase
        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples,
            seq_length=self.query_len,
            tokenizer=self.tokenizer
        )
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)

        # BERT reasons
        word_id_reason = []
        word_mask_reason = []
        for i in range(3):
            ex_r = read_examples(str(reasons[i]), idx)
            fea_r = convert_examples_to_features(
                examples=ex_r,
                seq_length=self.query_len,
                tokenizer=self.tokenizer
            )
            word_id_reason.append(fea_r[0].input_ids)
            word_mask_reason.append(fea_r[0].input_mask)

        target['word_id_reason_0'] = torch.tensor(word_id_reason[0], dtype=torch.long)
        target['word_mask_reason_0'] = torch.tensor(word_mask_reason[0], dtype=torch.bool)
        target['word_id_reason_1'] = torch.tensor(word_id_reason[1], dtype=torch.long)
        target['word_mask_reason_1'] = torch.tensor(word_mask_reason[1], dtype=torch.bool)
        target['word_id_reason_2'] = torch.tensor(word_id_reason[2], dtype=torch.long)
        target['word_mask_reason_2'] = torch.tensor(word_mask_reason[2], dtype=torch.bool)
        # print("target['word_id_reason_0']",target['word_id_reason_0'])
        if 'mask' in target:
            m = target.pop('mask')
            return image, m, target

        return image, target
