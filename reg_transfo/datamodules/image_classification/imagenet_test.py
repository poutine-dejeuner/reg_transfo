import pytest

from reg_transfo.conftest import setup_with_overrides
from reg_transfo.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from reg_transfo.datamodules.image_classification.imagenet import ImageNetDataModule
from reg_transfo.utils.testutils import needs_network_dataset_dir


@pytest.mark.slow
@needs_network_dataset_dir("imagenet")
@setup_with_overrides("datamodule=imagenet")
class TestImageNetDataModule(ImageClassificationDataModuleTests[ImageNetDataModule]): ...
