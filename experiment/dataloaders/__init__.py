"""API of dataloaders"""

from experiment.dataloaders.decoder import Decoder
from experiment.dataloaders.parser import Parser

from experiment.dataloaders.classification_input import ClassificationDecoder
from experiment.dataloaders.classification_input import ClassificationParser

from experiment.dataloaders.tfds_classification_decoders \
    import TfdsClassificationDecoder
from experiment.dataloaders.tfds_detection_decoders import COCODecoder
from experiment.dataloaders.tfds_segmentation_decoders import CityScapesDecorder

from experiment.dataloaders.input_reader import CombinationDatasetInputReader

from experiment.dataloaders.tf_example_decoder import TfExampleDecoder
from experiment.dataloaders.tf_example_label_map_decoder \
    import TfExampleDecoderLabelMap

from experiment.dataloaders.input_reader_factory import input_reader_generator
