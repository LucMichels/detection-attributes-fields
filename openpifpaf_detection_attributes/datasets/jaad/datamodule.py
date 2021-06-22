import argparse

import torch
import openpifpaf

from .attribute import JaadType
from .dataset import JaadDataset
from . import transforms
from .. import annotation
from .. import attribute
from .. import encoder
from .. import headmeta
from .. import metrics as eval_metrics
import sys


class Jaad(openpifpaf.datasets.DataModule):
    """DataModule for dataset JAAD."""

    debug = False
    pin_memory = False

    # General
    root_dir = 'data-jaad/'
    subset = 'default'
    train_set = 'train'
    val_set = 'val'
    test_set = 'test'

    # Tasks
    pedestrian_attributes = ['detection']
    occlusion_level = 1
    upsample_stride = 1

    # Pre-processing
    image_width = 961
    top_crop_ratio = 0.33
    image_height_stride = 16
    fast_scaling = True
    augmentation = True


    def __init__(self):
        super().__init__()
        self.compute_attributes()
        self.compute_head_metas()


    @classmethod
    def compute_attributes(cls):
        cls.attributes = {
            JaadType.PEDESTRIAN: cls.pedestrian_attributes,
        }


    @classmethod
    def compute_head_metas(cls):
        att_metas = attribute.get_attribute_metas(dataset='jaad',
                                                  attributes=cls.attributes)
        cls.head_metas = [headmeta.AttributeMeta('attribute-'+am['attribute'],
                                                 'jaad', **am)
                          for am in att_metas]
        for hm in cls.head_metas:
            hm.upsample_stride = cls.upsample_stride


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Jaad')

        # General
        group.add_argument('--jaad-root-dir',
                           default=cls.root_dir,
                           help='root directory of jaad dataset')
        group.add_argument('--jaad-subset',
                           default=cls.subset,
                           choices=['default', 'all_videos', 'high_visibility'],
                           help='subset of videos to consider')
        group.add_argument('--jaad-training-set',
                           default=cls.train_set,
                           choices=['train', 'trainval', 'hazik_train'],
                           help='training set')
        group.add_argument('--jaad-validation-set',
                           default=cls.val_set,
                           choices=['val', 'test', 'hazik_test'],
                           help='validation set')
        group.add_argument('--jaad-testing-set',
                           default=cls.test_set,
                           choices=['val', 'test', 'hazik_test'],
                           help='testing set')

        # Tasks
        group.add_argument('--jaad-pedestrian-attributes',
                           default=cls.pedestrian_attributes, nargs='+',
                           help='list of attributes to consider for pedestrians')
        group.add_argument('--jaad-occlusion-level',
                           default=cls.occlusion_level, type=int,
                           choices=[0, 1, 2],
                           help='max level of occlusion to learn from')
        group.add_argument('--jaad-head-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

        # Pre-processing
        group.add_argument('--jaad-image-width',
                           default=cls.image_width, type=int,
                           help='width to rescale image to')
        group.add_argument('--jaad-top-crop-ratio',
                           default=cls.top_crop_ratio, type=float,
                           help='ratio of height to crop from top of image')
        group.add_argument('--jaad-image-height-stride',
                           default=cls.image_height_stride, type=int,
                           help='stride to compute height of image')
        group.add_argument('--jaad-invert',
                           dest='jaad_invert', type=int,
                           default=0,
                           help='Invert label of crossing pedestrians x frames before they cross (eg: "30": pedestrians is considered crossing 30 frames before he actually does')
        
        assert cls.fast_scaling
        group.add_argument('--jaad-no-fast-scaling',
                           dest='jaad_fast_scaling',
                           default=True, action='store_false',
                           help='do not use fast scaling algorithm')
        assert cls.augmentation
        group.add_argument('--jaad-no-augmentation',
                           dest='jaad_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--jaad-use-hazik-augmentation',
                           dest='use_hazik_augmentation',
                           default=False, action='store_true',
                           help='apply hazik data augmentation. This overides any other augmentations')

        # Filtering
        group.add_argument('--jaad-truncate',
                           dest='jaad_truncate',
                           default=False, action='store_true',
                           help='on evaluation only consider  frames before crossing')
        group.add_argument('--jaad-slice',
                           dest='jaad_slice', nargs=2, type=int,
                           default=[0, 0],
                           help='on evaluation only consider a slice of frames. First argument is where to slice as the distance in frames where the pedestrian crosses \
                           (eg: "30": from 30 frames before he crosses, "-30": from 30 frames after he crosses ) \
                           Second argument is the size in frames of the slice (takes the x previous frames')


        # Metrics
        group.add_argument('--jaad-metrics',
                           dest='jaad_metrics', type=str,
                           default=['instance'],
                           help='Chose evaluation metric. Choose list of metrics from [hazik_instance, hazik_classification, classification, instance] \
                           in format metric1-metric2-...-metricn. Eg: instance-classification')




    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # General
        cls.root_dir = args.jaad_root_dir
        cls.subset = args.jaad_subset
        cls.train_set = args.jaad_training_set
        cls.val_set = args.jaad_validation_set
        cls.test_set = args.jaad_testing_set

        # Tasks
        cls.pedestrian_attributes = args.jaad_pedestrian_attributes
        cls.compute_attributes()
        cls.occlusion_level = args.jaad_occlusion_level
        cls.upsample_stride = args.jaad_head_upsample
        cls.compute_head_metas()

        # Pre-processing
        cls.image_width = args.jaad_image_width
        cls.top_crop_ratio = args.jaad_top_crop_ratio
        cls.image_height_stride = args.jaad_image_height_stride
        cls.fast_scaling = args.jaad_fast_scaling
        cls.augmentation = args.jaad_augmentation
        cls.use_hazik_augmentation = args.use_hazik_augmentation
        cls.invert = args.jaad_invert

        # Filtering
        cls.truncate = args.jaad_truncate
        cls.slice = args.jaad_slice

        # Metrics
        cls.metrics = args.jaad_metrics
        

    def _common_preprocess_op(self):
        return [
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(self.image_width,
                                       fast=self.fast_scaling),
            transforms.CropTopOut(self.top_crop_ratio,
                                  self.image_height_stride),
        ]


    def _train_preprocess(self):
        if self.use_hazik_augmentation:
            data_augmentation_op = [
                openpifpaf.transforms.RandomApply(transforms.HFlip(), 0.5),
                transforms.EVAL_TRANSFORM,
            ]
        elif self.augmentation:
            data_augmentation_op = [
                transforms.ZoomInOrOut(fast=self.fast_scaling),
                openpifpaf.transforms.RandomApply(transforms.HFlip(), 0.5),
                transforms.TRAIN_TRANSFORM,
            ]
        else:
            data_augmentation_op = [transforms.EVAL_TRANSFORM]

        encoders = [encoder.AttributeEncoder(
                        head_meta,
                        occlusion_level=self.occlusion_level,
                    )
                    for head_meta in self.head_metas]

        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            *data_augmentation_op,
            openpifpaf.transforms.Encoders(encoders),
        ])


    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            transforms.ToAnnotations(annotation.OBJECT_ANNOTATIONS['jaad']),
            transforms.EVAL_TRANSFORM,
        ])


    def train_loader(self):
        train_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.train_set,
            subset=self.subset,
            truncate=self.truncate,
            slice=self.slice,
            invert=self.invert,
            preprocess=self._train_preprocess(),
        )
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=(not self.debug) and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )


    def val_loader(self):
        val_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.val_set,
            subset=self.subset,
            truncate=self.truncate,
            slice=self.slice,
            invert=self.invert,
            preprocess=self._train_preprocess(),
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=(not self.debug) and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )


    def eval_loader(self):
        eval_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.test_set,
            subset=self.subset,
            truncate=self.truncate,
            slice=self.slice,
            invert=self.invert,
            preprocess=self._eval_preprocess(),
        )
        return torch.utils.data.DataLoader(
            eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta,
        )

    def keyword_to_metric(self, keyword):
        if keyword == "hazik_instance":
            return eval_metrics.InstanceHazikDetection(self.head_metas)
        elif keyword == "hazik_classification":
            return eval_metrics.ClassificationHazik(self.head_metas)
        elif keyword == "classification":
            return eval_metrics.Classification(self.head_metas)
        else: # default to Taylor metric
            return eval_metrics.InstanceDetection(self.head_metas)

    def metrics(self):
        chosen_metrics = self.metrics.split("-")
        if len(chosen_metrics) > 0:
            print(chosen_metrics)
            sys.stdout.flush()
            return [self.keyword_to_metric(chosen_metrics) in self.metrics]
        else:
            return [eval_metrics.InstanceDetection(self.head_metas)]
        
