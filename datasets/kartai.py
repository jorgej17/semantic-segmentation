import os
import os.path as path

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
import datasets.uniform as uniform

class Loader(BaseLoader):
    num_classes = 2
    trainid_to_name = {}
    ignore_label = -100

    color_mapping = []

    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality, mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        ######################################################################
        # kartai-specific stuff:
        ######################################################################
        self.root = cfg.DATASET.KARTAI_DIR
        img_ext = 'tif'
        mask_ext = 'tif'
        img_root = path.join(self.root, 'images')
        mask_root = path.join(self.root, 'labels')
        self.fill_colormap()


        self.all_imgs = self.find_kartai_images(img_root, mask_root, img_ext, mask_ext, mode)
        

        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)

        
        logx.msg(f'cn num_classes {self.num_classes}')
       
        self.build_epoch()
        
        
    def find_kartai_images(self, img_root, mask_root, img_ext,
                               mask_ext, run_mode):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        cities: a list of cities, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []
        img_dir = '{root}/{mode}'.format(root=img_root, mode = run_mode)
        mask_dir = '{root}/{mode}'.format(root=mask_root, mode = run_mode)

        print("IMGDIR", img_dir)
        for file_name in os.listdir(img_dir):
            basename, ext = os.path.splitext(file_name)
            assert ext == '.' + img_ext
            full_img_fn = os.path.join(img_dir, file_name)             
            full_mask_fn = os.path.join(mask_dir, file_name)
            
            items.append((full_img_fn, full_mask_fn))

        logx.msg('mode {} found {} images'.format(self.mode, len(items)))
        
        return items

    def fill_colormap(self):
        
        palette = [128, 64, 128, 244, 35, 232]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
            
        self.color_mapping = palette