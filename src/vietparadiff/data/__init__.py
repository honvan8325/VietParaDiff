"""Public dataset-building API."""

from vietparadiff.data.audit import (
    AuditIssue,
    DatasetAuditor,
    DatasetSnapshot,
    compute_dataset_snapshot,
)
from vietparadiff.data.cvl import build_cvl_dataset
from vietparadiff.data.iam import build_iam_dataset
from vietparadiff.data.splits import SplitConfig, create_data_splits
from vietparadiff.data.pipeline import (
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
from vietparadiff.data.uithwdb import build_uithwdb_dataset

__all__ = [
    "build_cvl_dataset",
    "build_iam_dataset",
    "build_uithwdb_dataset",
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
    "AuditIssue",
    "DatasetAuditor",
    "DatasetSnapshot",
    "compute_dataset_snapshot",
]
