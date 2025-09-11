# dataset/age_estimation_dataset_morph_lrc_dfe.py
from .age_estimation_dataset_lrc_dfe import AgeEstimationDatasetLrcDfe

class AgeEstimationDatasetMorphLrcDfe(AgeEstimationDatasetLrcDfe):
    def __init__(self, root_dir, split="val", **kw):
        super().__init__(root_dir=root_dir, dataset_name="MORPH", split=split, **kw)