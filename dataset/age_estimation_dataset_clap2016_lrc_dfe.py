# dataset/age_estimation_dataset_clap2016_lrc_dfe.py
from .age_estimation_dataset_lrc_dfe import AgeEstimationDatasetLrcDfe

class AgeEstimationDatasetCLAP2016LrcDfe(AgeEstimationDatasetLrcDfe):
    def __init__(self, root_dir, split="val", clap2016_csv=None, **kw):
        super().__init__(root_dir=root_dir, dataset_name="CLAP2016", split=split, clap2016_csv=clap2016_csv, **kw)