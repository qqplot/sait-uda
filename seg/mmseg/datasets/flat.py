from .builder import DATASETS
from .custom import CustomDataset

from . import FishDataset

@DATASETS.register_module()
class FlatDataset(CustomDataset):
    """Flat dataset.
    """

    CLASSES = FishDataset.CLASSES
    PALETTE = FishDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(FlatDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=None,
            **kwargs)
        
