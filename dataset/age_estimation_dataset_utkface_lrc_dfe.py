# dataset/age_estimation_dataset_utkface_lrc_dfe.py
from .age_estimation_dataset_lrc_dfe import AgeEstimationDatasetLrcDfe

class AgeEstimationDatasetUTKFaceLrcDfe(AgeEstimationDatasetLrcDfe):
    def __init__(self, root_dir, split="val", **kw):
        super().__init__(root_dir=root_dir, dataset_name="UTKFACE", split=split, **kw)