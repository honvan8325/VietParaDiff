"""Public dataset-building API."""

from src.data.cvl import build_cvl_dataset
from src.data.iam import build_iam_dataset
from src.data.splits import SplitConfig, create_data_splits
from src.data.training import (
    AutoKLDataset,
    HTRDataset,
    HTRImageProcessor,
    HTRVocabulary,
    HeightBucketBatchSampler,
    ParagraphImageProcessor,
    ReferenceImageProcessor,
    VietParaDiffCollator,
    VietParaDiffDataset,
    WidthBucketBatchSampler,
    collate_autokl,
    collate_htr,
)
from src.data.uithwdb import build_uithwdb_dataset
from src.data.vnondb import build_vnondb_dataset

__all__ = [
    "build_cvl_dataset",
    "build_iam_dataset",
    "build_uithwdb_dataset",
    "build_vnondb_dataset",
    "SplitConfig",
    "create_data_splits",
    "ParagraphImageProcessor",
    "HTRImageProcessor",
    "ReferenceImageProcessor",
    "HTRVocabulary",
    "AutoKLDataset",
    "HTRDataset",
    "VietParaDiffDataset",
    "HeightBucketBatchSampler",
    "WidthBucketBatchSampler",
    "collate_autokl",
    "collate_htr",
    "VietParaDiffCollator",
]
