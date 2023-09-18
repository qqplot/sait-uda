import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class FishDataset(CustomDataset):
    """Fish dataset.
    """
    CLASSES = ('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole', 'Traffic Light', 'Traffic Sign', 'Nature', 'Sky', 'Person', 'Rider', 'Car', 'Background')
    PALETTE = [[0, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 0],
            [0, 0, 128], [128, 128, 128], [128, 0, 0], [128, 0, 128], [0, 0, 0]]

    def __init__(self,
                 crop_pseudo_margins=None,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super(FishDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [1024, 2048]

    def pre_pipeline(self, results):
        super(FishDataset, self).pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')