import argparse
import logging
import time
from typing import List
import sys
import cv2
import numpy as np
import openpifpaf
from scipy.special import softmax

from .. import optics
from ...datasets import annotation
from ...datasets import attribute
from ...datasets import headmeta

from . import cifcaf_threadless


LOG = logging.getLogger(__name__)


class InstanceDecoder(openpifpaf.decoder.decoder.Decoder):
    """Decoder to convert predicted fields to sets of instance detections.

    Args:
        dataset (str): Dataset name.
        object_type (ObjectType): Type of object detected.
        attribute_metas (List[AttributeMeta]): List of meta information about
            predicted attributes.
    """

    # General
    dataset = None
    object_type = None

    # Clustering detections
    s_threshold = 0.2
    optics_min_cluster_size = 10
    optics_epsilon = 5.0
    optics_cluster_threshold = 0.5


    def __init__(self,
                 dataset: str,
                 object_type: attribute.ObjectType,
                 attribute_metas: List[headmeta.AttributeMeta]):
        super().__init__()
        self.dataset = dataset
        self.object_type = object_type
        self.annotation = annotation.OBJECT_ANNOTATIONS[self.dataset][self.object_type]
        for meta in attribute_metas:
            assert meta.dataset == self.dataset
            assert meta.object_type is self.object_type
        self.attribute_metas = attribute_metas


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('InstanceDecoder')

        # Clustering detections
        group.add_argument('--decoder-s-threshold',
                           default=cls.s_threshold, type=float,
                           help='threshold for field S')
        group.add_argument('--decoder-optics-min-cluster-size',
                           default=cls.optics_min_cluster_size, type=int,
                           help='minimum size of clusters in OPTICS')
        group.add_argument('--decoder-optics-epsilon',
                           default=cls.optics_epsilon, type=float,
                           help='maximum radius of cluster in OPTICS')
        group.add_argument('--decoder-optics-cluster-threshold',
                           default=cls.optics_cluster_threshold, type=float,
                           help='threshold to separate clusters in OPTICS')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Clustering detections
        cls.s_threshold = args.decoder_s_threshold
        cls.optics_min_cluster_size = args.decoder_optics_min_cluster_size
        cls.optics_epsilon = args.decoder_optics_epsilon
        cls.optics_cluster_threshold = args.decoder_optics_cluster_threshold


    @classmethod
    def factory(self, head_metas: List[openpifpaf.headmeta.Base]):
        decoders = []
        for dataset in attribute.OBJECT_TYPES:
            for object_type in attribute.OBJECT_TYPES[dataset]:
                meta_list = [meta for meta in head_metas
                             if (
                                isinstance(meta, headmeta.AttributeMeta)
                                and (meta.dataset == dataset)
                                and (meta.object_type is object_type)
                             )]
                if len(meta_list) > 0:
                    decoders.append(InstanceDecoder(dataset=dataset,
                                                    object_type=object_type,
                                                    attribute_metas=meta_list))
        return decoders


    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        # Field S
        s_meta = [meta for meta in self.attribute_metas
                  if meta.attribute == 'confidence']
        assert len(s_meta) == 1
        s_meta = s_meta[0]
        s_field = fields[s_meta.head_index].copy()
        conf_field = 1. / (1. + np.exp(-s_field))
        s_mask = conf_field > self.s_threshold

        # Field V
        v_meta = [meta for meta in self.attribute_metas
                  if meta.attribute == 'center']
        assert len(v_meta) == 1
        v_meta = v_meta[0]
        v_field = fields[v_meta.head_index].copy()
        if v_meta.std is not None:
            v_field[0] *= v_meta.std[0]
            v_field[1] *= v_meta.std[1]
        if v_meta.mean is not None:
            v_field[0] += v_meta.mean[0]
            v_field[1] += v_meta.mean[1]

        # OPTICS clustering
        point_list = []
        for y in range(s_mask.shape[1]):
            for x in range(s_mask.shape[2]):
                if s_mask[0,y,x]:
                    point = optics.Point(x, y, v_field[0,y,x], v_field[1,y,x])
                    point_list.append(point)

        clustering = optics.Optics(point_list,
                                   self.optics_min_cluster_size,
                                   self.optics_epsilon)
        clustering.run()
        clusters = clustering.cluster(self.optics_cluster_threshold)

        # Predictions for all instances
        predictions = []
        for cluster in clusters:
            attributes = {}
            for meta in self.attribute_metas:
                att = self.cluster_vote(fields[meta.head_index], cluster,
                                        meta, conf_field)
                attributes[meta.attribute] = att

            pred = self.annotation(**attributes)
            predictions.append(pred)

        LOG.info('predictions %d, %.3fs',
                  len(predictions), time.perf_counter()-start)

        return predictions


    def cluster_vote(self, field, cluster, meta, conf_field):
        field = field.copy()

        if meta.std is not None:
            field *= (meta.std if meta.n_channels == 1
                      else np.expand_dims(meta.std, (1,2)))
        if meta.mean is not None:
            field += (meta.mean if meta.n_channels == 1
                      else np.expand_dims(meta.mean, (1,2)))

        pred = np.array([0.]*field.shape[0])
        norm = 0.
        for pt in cluster.points:
            if meta.is_scalar: # scalar field
                val = field[:, pt.y, pt.x]
            else: # vectorial field
                val = np.array([pt.x, pt.y]) + field[:, pt.y, pt.x]
            conf = (
                conf_field[0, pt.y, pt.x] if meta.attribute != 'confidence'
                else 1.
            )
            pred += val * conf
            norm += conf
        pred = pred / norm if norm != 0. else 0.

        if meta.is_spatial:
            pred *= meta.stride
        if meta.n_channels == 1:
            if meta.is_classification:
                pred = 1. / (1. + np.exp(-pred))
            pred = pred[0]
        else:
            if meta.is_classification:
                pred = softmax(pred)
            pred = pred.tolist()

        return pred





class InstanceHazikCIFCAFDecoder(openpifpaf.decoder.decoder.Decoder):
    """Decoder to convert predicted fields to sets of instance detections.

    Args:
        dataset (str): Dataset name.
        object_type (ObjectType): Type of object detected.
        attribute_metas (List[AttributeMeta]): List of meta information about
            predicted attributes.
        full_head_metas (List[openpifpaf.headmeta.Base]): Full list of meta information about all heads used by the network.

    """

    # General
    dataset = None
    object_type = None

    def __init__(self,
                 dataset: str,
                 object_type: attribute.ObjectType,
                 attribute_metas: List[headmeta.AttributeMeta],
                 full_head_metas: List[openpifpaf.headmeta.Base]):
        super().__init__()
        self.dataset = dataset
        self.object_type = object_type
        self.annotation = annotation.OBJECT_ANNOTATIONS[self.dataset][self.object_type]
        for meta in attribute_metas:
            assert meta.dataset == self.dataset
            assert meta.object_type is self.object_type
        self.attribute_metas = attribute_metas
        self.full_head_metas = full_head_metas

    @classmethod
    def factory(self, head_metas: List[openpifpaf.headmeta.Base]):
        decoders = []
        for dataset in attribute.OBJECT_TYPES:
            for object_type in attribute.OBJECT_TYPES[dataset]:
                meta_list = [meta for meta in head_metas
                             if (
                                isinstance(meta, headmeta.AttributeMeta)
                                and (meta.dataset == dataset)
                                and (meta.object_type is object_type)
                             )]
                if len(meta_list) > 0:
                    decoders.append(InstanceHazikCIFCAFDecoder(dataset=dataset,
                                                    object_type=object_type,
                                                    attribute_metas=meta_list,
                                                    full_head_metas=head_metas))

        return decoders

    def __call__(self, fields, initial_annotations=None):

        start = time.perf_counter()

        assert len(fields) >= len(self.attribute_metas) + 2 # make sure we have enough kept all fields (--head-consolidation=keep)
        cif_head = [meta for meta in self.full_head_metas if isinstance(meta, openpifpaf.headmeta.Cif)]
        caf_head = [meta for meta in self.full_head_metas if isinstance(meta, openpifpaf.headmeta.Caf)]
        assert len(cif_head) == len(caf_head) and len(caf_head) == 1 # make sure we have the openpifpaf heads (model trained with cocokp and the cifcaf heads)
        cifcaf_dec = cifcaf_threadless.CifCaf(cif_head, caf_head)
        parser = argparse.ArgumentParser()
        cifcaf_dec.cli(parser)
        args, _ = parser.parse_known_args()
        cifcaf_dec.configure(args)
        annotations_cifcaf = cifcaf_dec(fields)

        predictions = []
        for ann in annotations_cifcaf:

            if ann.score > 0:
                bbox = ann.bbox()

                attributes = {}

                c, w, h = self.get_center_width_height_from(bbox)
                attributes["center"] = c
                attributes["width"]  = w
                attributes["height"] = h
                attributes["confidence"] = ann.score

                if ann.score > 0:
                    for meta in self.attribute_metas:
                        att = self.bbox_vote(fields[meta.head_index], bbox, meta)
                        attributes[meta.attribute] = att

                    pred = self.annotation(**attributes)
                    predictions.append(pred)
                
        LOG.info('predictions %d, %.3fs',
                  len(predictions), time.perf_counter()-start)
        return predictions

    def bbox_vote(self, field, bbox, meta):
        field = field.copy()

        # rescale bbox so its fit fields
        #bbox = [val/(meta.base_stride/meta.upsample_stride) for val in bbox] 


        bbox = np.round(bbox).astype(np.int)
        w = max(1, bbox[2])
        h = max(1, bbox[3])
        x = bbox[0] 
        y = bbox[1]

        field = field.squeeze(0) * 255
        field = cv2.resize(field,
             (int(field.shape[1]*(meta.base_stride/meta.upsample_stride)),
             int(field.shape[0]*(meta.base_stride/meta.upsample_stride)))
             )

        # generate the distribution centered at this box
        x0, y0, sigma_x, sigma_y = x+float(w)/2, y+float(h)/2, float(w)/4, float(h)/4

        # activity map for current person
        y, x = np.arange(field.shape[0]), np.arange(field.shape[1])    
        gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
        gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
        g  = np.outer(gy, gx)

        pred = np.sum(g*field)


        return pred

    def get_center_width_height_from(self, bbox):
        w = bbox[2]
        h = bbox[3]
        x = bbox[0] 
        y = bbox[1]
        c = [x + 0.5*w, y + 0.5*h]
        return c, w, h

class InstanceCIFCAFDecoder(openpifpaf.decoder.decoder.Decoder):
    """Decoder to convert predicted fields to sets of instance detections.

    Args:
        dataset (str): Dataset name.
        object_type (ObjectType): Type of object detected.
        attribute_metas (List[AttributeMeta]): List of meta information about
            predicted attributes.
        full_head_metas (List[openpifpaf.headmeta.Base]): Full list of meta information about all heads used by the network.

    """

    # General
    dataset = None
    object_type = None

    # Clustering detections
    s_threshold = 0.2
    optics_min_cluster_size = 10
    optics_epsilon = 5.0
    optics_cluster_threshold = 0.5

    # pedestrian detection
    decoder_use_pifpaf_bbox = False

    def __init__(self,
                 dataset: str,
                 object_type: attribute.ObjectType,
                 attribute_metas: List[headmeta.AttributeMeta],
                 full_head_metas: List[openpifpaf.headmeta.Base]):
        super().__init__()
        self.dataset = dataset
        self.object_type = object_type
        self.annotation = annotation.OBJECT_ANNOTATIONS[self.dataset][self.object_type]
        for meta in attribute_metas:
            assert meta.dataset == self.dataset
            assert meta.object_type is self.object_type
        self.attribute_metas = attribute_metas
        self.full_head_metas = full_head_metas


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('InstanceCIFCAFDecoder')

        # Clustering detections
        group.add_argument('--decoder-s-threshold',
                           default=cls.s_threshold, type=float,
                           help='threshold for field S')
        group.add_argument('--decoder-optics-min-cluster-size',
                           default=cls.optics_min_cluster_size, type=int,
                           help='minimum size of clusters in OPTICS')
        group.add_argument('--decoder-optics-epsilon',
                           default=cls.optics_epsilon, type=float,
                           help='maximum radius of cluster in OPTICS')
        group.add_argument('--decoder-optics-cluster-threshold',
                           default=cls.optics_cluster_threshold, type=float,
                           help='threshold to separate clusters in OPTICS')
        group.add_argument('--decoder-use-pifpaf-bbox',
                   default=False, action='store_true',
                   help='use pifpaf head bboxes from keypoints instead of default detection heads')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Clustering detections
        cls.s_threshold = args.decoder_s_threshold
        cls.optics_min_cluster_size = args.decoder_optics_min_cluster_size
        cls.optics_epsilon = args.decoder_optics_epsilon
        cls.optics_cluster_threshold = args.decoder_optics_cluster_threshold

        # pifpaf detection
        cls.decoder_use_pifpaf_bbox = args.decoder_use_pifpaf_bbox

    @classmethod
    def factory(self, head_metas: List[openpifpaf.headmeta.Base]):
        decoders = []
        for dataset in attribute.OBJECT_TYPES:
            for object_type in attribute.OBJECT_TYPES[dataset]:
                meta_list = [meta for meta in head_metas
                             if (
                                isinstance(meta, headmeta.AttributeMeta)
                                and (meta.dataset == dataset)
                                and (meta.object_type is object_type)
                             )]
                if len(meta_list) > 0:
                    decoders.append(InstanceCIFCAFDecoder(dataset=dataset,
                                                    object_type=object_type,
                                                    attribute_metas=meta_list,
                                                    full_head_metas=head_metas))

        return decoders



    def __call__(self, fields, initial_annotations=None):

        start = time.perf_counter()
        if not self.decoder_use_pifpaf_bbox:
            # Field S
            s_meta = [meta for meta in self.attribute_metas
                      if meta.attribute == 'confidence']
            assert len(s_meta) == 1
            s_meta = s_meta[0]
            s_field = fields[s_meta.head_index].copy()
            conf_field = 1. / (1. + np.exp(-s_field))
            s_mask = conf_field > self.s_threshold

            # Field V
            v_meta = [meta for meta in self.attribute_metas
                      if meta.attribute == 'center']
            assert len(v_meta) == 1
            v_meta = v_meta[0]
            v_field = fields[v_meta.head_index].copy()
            if v_meta.std is not None:
                v_field[0] *= v_meta.std[0]
                v_field[1] *= v_meta.std[1]
            if v_meta.mean is not None:
                v_field[0] += v_meta.mean[0]
                v_field[1] += v_meta.mean[1]

            # OPTICS clustering
            point_list = []
            for y in range(s_mask.shape[1]):
                for x in range(s_mask.shape[2]):
                    if s_mask[0,y,x]:
                        point = optics.Point(x, y, v_field[0,y,x], v_field[1,y,x])
                        point_list.append(point)

            clustering = optics.Optics(point_list,
                                       self.optics_min_cluster_size,
                                       self.optics_epsilon)
            clustering.run()
            clusters = clustering.cluster(self.optics_cluster_threshold)

            # Predictions for all instances
            predictions = []
            for cluster in clusters:
                attributes = {}
                for meta in self.attribute_metas:
                    att = self.cluster_vote(fields[meta.head_index], cluster,
                                            meta, conf_field)
                    attributes[meta.attribute] = att

                pred = self.annotation(**attributes)
                predictions.append(pred)

            LOG.info('predictions %d, %.3fs',
                      len(predictions), time.perf_counter()-start)

        else:

            assert len(fields) >= len(self.attribute_metas) + 2 # make sure we have enough kept all fields (--head-consolidation=keep)
            cif_head = [meta for meta in self.full_head_metas if isinstance(meta, openpifpaf.headmeta.Cif)]
            caf_head = [meta for meta in self.full_head_metas if isinstance(meta, openpifpaf.headmeta.Caf)]
            assert len(cif_head) == len(caf_head) and len(caf_head) == 1 # make sure we have the openpifpaf heads (model trained with cocokp and the cifcaf heads)
            cifcaf_dec = cifcaf_threadless.CifCaf(cif_head, caf_head)
            parser = argparse.ArgumentParser()
            cifcaf_dec.cli(parser)
            args, _ = parser.parse_known_args()
            cifcaf_dec.configure(args)
            annotations_cifcaf = cifcaf_dec(fields)

            predictions = []
            for ann in annotations_cifcaf:

                if ann.score > 0:
                    bbox = ann.bbox()

                    attributes = {}

                    c, w, h = self.get_center_width_height_from(bbox)
                    attributes["center"] = c
                    attributes["width"]  = w
                    attributes["height"] = h
                    attributes["confidence"] = ann.score

                    # for now we remove detections that are too small but we might try with setting these to 1
                    if ann.score > 0:
                        for meta in self.attribute_metas:
                            att = self.bbox_vote(fields[meta.head_index], bbox, meta)
                            attributes[meta.attribute] = att

                        pred = self.annotation(**attributes)
                        predictions.append(pred)
                    
            LOG.info('predictions %d, %.3fs',
                      len(predictions), time.perf_counter()-start)
        return predictions

    def cluster_vote(self, field, cluster, meta, conf_field):
        field = field.copy()

        if meta.std is not None:
            field *= (meta.std if meta.n_channels == 1
                      else np.expand_dims(meta.std, (1,2)))
        if meta.mean is not None:
            field += (meta.mean if meta.n_channels == 1
                      else np.expand_dims(meta.mean, (1,2)))

        pred = np.array([0.]*field.shape[0])
        norm = 0.
        for pt in cluster.points:
            if meta.is_scalar: # scalar field
                val = field[:, pt.y, pt.x]
            else: # vectorial field
                val = np.array([pt.x, pt.y]) + field[:, pt.y, pt.x]
            conf = (
                conf_field[0, pt.y, pt.x] if meta.attribute != 'confidence'
                else 1.
            )
            pred += val * conf
            norm += conf
        pred = pred / norm if norm != 0. else 0.

        if meta.is_spatial:
            pred *= meta.stride
        if meta.n_channels == 1:
            if meta.is_classification:
                pred = 1. / (1. + np.exp(-pred))
            pred = pred[0]
        else:
            if meta.is_classification:
                pred = softmax(pred)
            pred = pred.tolist()

        return pred

    def bbox_vote(self, field, bbox, meta):
        field = field.copy()

        assert meta.is_classification # rest is not implemented
        # rescale bbox so its fit fields
        bbox = [val/(meta.base_stride/meta.upsample_stride) for val in bbox]      
        bbox = np.round(bbox).astype(np.int)
        w = max(1, bbox[2])
        h = max(1, bbox[3])
        x = bbox[0] 
        y = bbox[1]

        field = field.squeeze(0)
        # generate the distribution centered at this box
        x0, y0, sigma_x, sigma_y = x+float(w)/2, y+float(h)/2, float(w)/4, float(h)/4

        # activity map for current person
        y, x = np.arange(field.shape[0]), np.arange(field.shape[1])    
        gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
        gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
        g  = np.outer(gy, gx)

        pred = np.sum(g*field)/sum(g)

        pred = 1. / (1. + np.exp(-pred))

        return pred



    def get_center_width_height_from(self, bbox):
        w = bbox[2]
        h = bbox[3]
        x = bbox[0] 
        y = bbox[1]
        c = [x + 0.5*w, y + 0.5*h]
        return c, w, h