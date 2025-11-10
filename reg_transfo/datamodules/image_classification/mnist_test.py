from reg_transfo.conftest import setup_with_overrides
from reg_transfo.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from reg_transfo.datamodules.image_classification.mnist import MNISTDataModule


@setup_with_overrides("datamodule=mnist")
class TestMNISTDataModule(ImageClassificationDataModuleTests[MNISTDataModule]): ...
