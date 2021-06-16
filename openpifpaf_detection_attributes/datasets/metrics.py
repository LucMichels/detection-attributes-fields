import copy
import json
import logging
import math
from typing import List
import zipfile
import sys
import openpifpaf
from numpy import argmax
from .headmeta import AttributeMeta

from .jaad.annotation import JaadPedestrianAnnotation

import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.special import softmax
LOG = logging.getLogger(__name__)


def compute_iou(pred_c, pred_w, pred_h, gt_c, gt_w, gt_h):
    inter_box = [
        max(pred_c[0] - .5*pred_w, gt_c[0] - .5*gt_w),
        max(pred_c[1] - .5*pred_h, gt_c[1] - .5*gt_h),
        min(pred_c[0] + .5*pred_w, gt_c[0] + .5*gt_w),
        min(pred_c[1] + .5*pred_h, gt_c[1] + .5*gt_h)
    ]
    inter_area = (
        max(0., inter_box[2] - inter_box[0])
        * max(0., inter_box[3] - inter_box[1])
    )
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    iou = (
        inter_area / (pred_area + gt_area - inter_area)
        if pred_area + gt_area - inter_area != 0 else 0.
    )
    return iou


def compute_ap(stats):
    tps = [tp for _, tp in sorted(zip(stats['score'],
                                      stats['tp']),
                                  key=lambda pair: pair[0],
                                  reverse=True)]
    fps = [fp for _, fp in sorted(zip(stats['score'],
                                      stats['fp']),
                                  key=lambda pair: pair[0],
                                  reverse=True)]
    cumsum = 0
    for idx, val in enumerate(tps):
        tps[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(fps):
        fps[idx] += cumsum
        cumsum += val
    recs = tps[:]
    for idx, val in enumerate(tps):
        recs[idx] = (
            float(tps[idx]) / stats['n_gt']
            if stats['n_gt'] != 0 else 0.
        )
    precs = tps[:]
    for idx, val in enumerate(tps):
        precs[idx] = (
            float(tps[idx]) / (tps[idx] + fps[idx])
            if tps[idx] + fps[idx] != 0 else 0.
        )
    return average_precision(recs, precs)

def compute_precision(stats):
    return precision_score(stats['true'], stats['pred'], average=None, labels=[1])[0]

def compute_recall(stats):
    return recall_score(stats['true'], stats['pred'], average=None, labels=[1])[0]

def average_precision(rec, prec):
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap

class InstanceDetection(openpifpaf.metric.base.Base):
    """Compute detection metrics from all detected instances for a list of
        attributes.
    Args:
        attribute_metas (List[AttributeMeta]): list of meta information about
            attributes.
    """

    def __init__(self, attribute_metas: List[AttributeMeta]):
        self.attribute_metas = [am for am in attribute_metas
                                if ((am.attribute == 'confidence')
                                    or (am.group != 'detection'))]
        assert len(self.attribute_metas) > 0

        self.det_stats = {}
        for att_meta in self.attribute_metas:
            if att_meta.is_classification:
                n_classes = max(att_meta.n_channels, 2)
            else:
                n_classes = 10
            self.det_stats[att_meta.attribute] = {'n_classes': n_classes}
            for cls in range(n_classes):
                self.det_stats[att_meta.attribute][cls] = {
                    'n_gt': 0, 'score': [], 'tp': [], 'fp': []}
        self.predictions = {}


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # Store predictions for writing to file
        pred_data = []
        predictions = [pred for pred in predictions if isinstance(pred, JaadPedestrianAnnotation)]
        for pred in predictions:
            pred_data.append(pred.json_data())
        self.predictions[image_meta['image_id']] = pred_data

        # Compute metrics
        for att_meta in self.attribute_metas:
            self.accumulate_attribute(att_meta, predictions, image_meta,
                                      ground_truth=ground_truth)


    def accumulate_attribute(self, attribute_meta, predictions, image_meta, *,
                             ground_truth=None):
        for cls in range(self.det_stats[attribute_meta.attribute]['n_classes']):
            det_stats = self.det_stats[attribute_meta.attribute][cls]

            # Initialize ground truths
            gt_match = {}
            for gt in ground_truth:
                if (
                    gt.ignore_eval
                    or (gt.attributes[attribute_meta.attribute] is None)
                    or (not attribute_meta.is_classification)
                    or (int(gt.attributes[attribute_meta.attribute]) == cls)
                ):
                    gt_match[gt.id] = False
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[attribute_meta.attribute] is not None)
                    ):
                        det_stats['n_gt'] += 1

            # Rank predictions based on confidences
            ranked_preds = []
            for pred in predictions:
                if (
                    (attribute_meta.attribute in pred.attributes)
                    and (pred.attributes[attribute_meta.attribute] is not None)
                ):
                    rpred = copy.deepcopy(pred)
                    pred_score = pred.attributes[attribute_meta.attribute]
                    pred_conf = pred.attributes['confidence']
                    if (
                        (attribute_meta.attribute == 'confidence')
                        or (not attribute_meta.is_classification)
                    ):
                        rpred.attributes['score'] = pred_conf
                    elif (
                        attribute_meta.is_classification
                        and (attribute_meta.n_channels == 1)
                    ):
                        rpred.attributes['score'] = (
                            (cls*pred_score + (1-cls)*(1.-pred_score))
                            * pred_conf
                        )
                    elif (
                        attribute_meta.is_classification
                        and (attribute_meta.n_channels > 1)
                    ):
                        rpred.attributes['score'] = pred_score[cls] * pred_conf
                    ranked_preds.append(rpred)
            ranked_preds.sort(key=lambda x:x.attributes['score'], reverse=True)

            # Match predictions with closest groud truths
            for pred in ranked_preds:
                max_iou = -1.
                match = None
                for gt in ground_truth:
                    if (
                        (gt.id in gt_match)
                        and ('width' in pred.attributes)
                        and ('height' in pred.attributes)
                    ):
                        iou = compute_iou(pred.attributes['center'], pred.attributes['width'],
                                          pred.attributes['height'],
                                          gt.attributes['center'], gt.attributes['width'],
                                          gt.attributes['height'])
                    else:
                        iou = 0.
                    if (iou > 0.5) and (iou >= max_iou):
                        if (
                            (gt.attributes[attribute_meta.attribute] is None)
                            or attribute_meta.is_classification
                            or (abs(gt.attributes[attribute_meta.attribute]
                                -pred.attributes[attribute_meta.attribute]) <= (cls+1)*.5)
                        ):
                            max_iou = iou
                            match = gt

                # Classify predictions as True Positives or False Positives
                if match is not None:
                    if (
                        (not match.ignore_eval)
                        and (match.attributes[attribute_meta.attribute] is not None)
                    ):
                        if not gt_match[match.id]:
                            # if pred.attributes[attribute_meta.attribute] != match.attributes[attribute_meta.attribute]:
                            #     print(pred.attributes[attribute_meta.attribute] != match.attributes[attribute_meta.attribute])
                            #     sys.stdout.flush()
                            # True positive
                            det_stats['score'].append(pred.attributes['score'])
                            det_stats['tp'].append(1)
                            det_stats['fp'].append(0)

                            gt_match[match.id] = True
                        else:
                            # False positive (multiple detections)
                            det_stats['score'].append(pred.attributes['score'])
                            det_stats['tp'].append(0)
                            det_stats['fp'].append(1)
                    else:
                        # Ignore instance
                        pass
                else:
                    # False positive
                    det_stats['score'].append(pred.attributes['score'])
                    det_stats['tp'].append(0)
                    det_stats['fp'].append(1)


    def stats(self):
        text_labels = []
        stats = []

        att_aps = []
        for att_meta in self.attribute_metas:
            cls_aps = []
            for cls in range(self.det_stats[att_meta.attribute]['n_classes']):
                cls_ap = compute_ap(self.det_stats[att_meta.attribute][cls])
                cls_aps.append(cls_ap)
            if att_meta.attribute == 'confidence':
                text_labels.append('detection_AP')
                stats.append(cls_aps[1])
                att_aps.append(cls_aps[1])
                LOG.info('detection AP = {}'.format(cls_aps[1]*100))
            else:
                text_labels.append(att_meta.attribute + '_AP')
                att_ap = sum(cls_aps) / len(cls_aps)
                stats.append(att_ap)
                att_aps.append(att_ap)
                LOG.info('{} AP = {}'.format(att_meta.attribute, att_ap*100))
        text_labels.append('attribute_mAP')
        map = sum(att_aps) / len(att_aps)
        stats.append(map)
        LOG.info('attribute mAP = {}'.format(map*100))

        data = {
            'text_labels': text_labels,
            'stats': stats,
        }
        return data


    def write_predictions(self, filename, *, additional_data=None):
        with open(filename + '.pred.json', 'w') as f:
            json.dump(self.predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)

class Classification(openpifpaf.metric.base.Base):
    """Compute detection metrics from all detected instances for a list of
        attributes.
    Args:
        attribute_metas (List[AttributeMeta]): list of meta information about
            attributes.
    """

    def __init__(self, attribute_metas: List[AttributeMeta]):
        self.attribute_metas = [am for am in attribute_metas
                                if ((am.attribute == 'confidence')
                                    or (am.group != 'detection'))]
        assert len(self.attribute_metas) > 0

        self.det_stats = {}
        for att_meta in self.attribute_metas:
            if att_meta.is_classification:
                n_classes = max(att_meta.n_channels, 2)
            else:
                n_classes = 10
            self.det_stats[att_meta.attribute] = {'n_classes': n_classes}
            for cls in range(n_classes):
                self.det_stats[att_meta.attribute][cls] = {
                    'n_gt': 0, 'score': [], 'tp': [], 'fp': []}
        self.predictions = {}


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # Store predictions for writing to file
        pred_data = []
        predictions = [pred for pred in predictions if isinstance(pred, JaadPedestrianAnnotation)]
        for pred in predictions:
            pred_data.append(pred.json_data())
        self.predictions[image_meta['image_id']] = pred_data

        # Compute metrics
        for att_meta in self.attribute_metas:
            self.accumulate_attribute(att_meta, predictions, image_meta,
                                      ground_truth=ground_truth)


    def accumulate_attribute(self, attribute_meta, predictions, image_meta, *,
                             ground_truth=None):

        ground_truth = [gt for gt in ground_truth if not gt.ignore_eval] # remove ground truth ignore eval as we loop over ground truth
        for cls in range(self.det_stats[attribute_meta.attribute]['n_classes']):
            det_stats = self.det_stats[attribute_meta.attribute][cls]

            # Initialize ground truths
            gt_match = {}
            for gt in ground_truth:
                if (
                    gt.ignore_eval
                    or (gt.attributes[attribute_meta.attribute] is None)
                    or (not attribute_meta.is_classification)
                    or (int(gt.attributes[attribute_meta.attribute]) == cls)
                ):
                    gt_match[gt.id] = False
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[attribute_meta.attribute] is not None)
                    ):
                        det_stats['n_gt'] += 1

            # Rank predictions based on confidences
            ranked_preds = []
            for pred in predictions:
                if (
                    (attribute_meta.attribute in pred.attributes)
                    and (pred.attributes[attribute_meta.attribute] is not None)
                ):
                    rpred = copy.deepcopy(pred)
                    pred_score = pred.attributes[attribute_meta.attribute]
                    pred_conf = pred.attributes['confidence']
                    if (
                        (attribute_meta.attribute == 'confidence')
                        or (not attribute_meta.is_classification)
                    ):
                        rpred.attributes['score'] = pred_conf
                    elif (
                        attribute_meta.is_classification
                        and (attribute_meta.n_channels == 1)
                    ):
                        rpred.attributes['score'] = (
                            (cls*pred_score + (1-cls)*(1.-pred_score))
                            * pred_conf
                        )
                    elif (
                        attribute_meta.is_classification
                        and (attribute_meta.n_channels > 1)
                    ):
                        rpred.attributes['score'] = pred_score[cls] * pred_conf
                    ranked_preds.append(rpred)
            ranked_preds.sort(key=lambda x:x.attributes['score'], reverse=True)

            # Match ground truths with closest predictions
            for gt in ground_truth:
                max_iou = -1.
                match = None
                for pred in ranked_preds:
                    if ((gt.id in gt_match)
                        and ('width' in pred.attributes)
                        and ('height' in pred.attributes)
                    ):
                        iou = compute_iou(pred.attributes['center'], pred.attributes['width'],
                                          pred.attributes['height'],
                                          gt.attributes['center'], gt.attributes['width'],
                                          gt.attributes['height'])
                    else:
                        iou = 0.
                    if (iou > 0.0) and (iou >= max_iou):
                        if (
                            (gt.attributes[attribute_meta.attribute] is None)
                            or attribute_meta.is_classification
                            or (abs(gt.attributes[attribute_meta.attribute]
                                -pred.attributes[attribute_meta.attribute]) <= (cls+1)*.5)
                        ):
                            max_iou = iou
                            match = pred

                # Classify predictions as True Positives or False Positives
                if match is not None:
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[attribute_meta.attribute] is not None)
                    ):
                        # True positive
                        det_stats['score'].append(match.attributes['score'])
                        det_stats['tp'].append(1)
                        det_stats['fp'].append(0)

                    else:
                        # Ignore instance
                        pass
                else:
                    # False positive
                    det_stats['score'].append(0)
                    det_stats['tp'].append(0)
                    det_stats['fp'].append(1)


    def stats(self):
        text_labels = []
        stats = []

        att_aps = []
        for att_meta in self.attribute_metas:
            cls_aps = []
            for cls in range(self.det_stats[att_meta.attribute]['n_classes']):
                cls_ap = compute_ap(self.det_stats[att_meta.attribute][cls])
                cls_aps.append(cls_ap)
            if att_meta.attribute == 'confidence':
                text_labels.append('detection_AP')
                stats.append(cls_aps[1])
                att_aps.append(cls_aps[1])
                LOG.info('detection AP = {}'.format(cls_aps[1]*100))
            else:
                text_labels.append(att_meta.attribute + '_AP')
                att_ap = sum(cls_aps) / len(cls_aps)
                stats.append(att_ap)
                att_aps.append(att_ap)
                LOG.info('{} AP = {}'.format(att_meta.attribute, att_ap*100))
        text_labels.append('attribute_mAP')
        map = sum(att_aps) / len(att_aps)
        stats.append(map)
        LOG.info('attribute mAP = {}'.format(map*100))

        data = {
            'text_labels': text_labels,
            'stats': stats,
        }
        return data


    def write_predictions(self, filename, *, additional_data=None):
        with open(filename + '.pred.json', 'w') as f:
            json.dump(self.predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)


class InstanceHazikDetection(openpifpaf.metric.base.Base):
    """Compute classification metrics from all ground truth instances one channel classification attributes

    Args:
        attribute_metas (List[AttributeMeta]): list of meta information about
            attributes.
    """

    def __init__(self, attribute_metas: List[AttributeMeta]):
        self.attribute_metas = [am for am in attribute_metas
                                if ((am.attribute == "is_crossing_reg")
                                or (am.attribute == "is_not_crossing_reg"))]

        assert len(self.attribute_metas) == 2 # only valid classification attributes (is_crossing and is_not_crossing) are allowed
        self.cros_stats = {}
        for att in ["is_crossing", "is_not_crossing"]:
            self.cros_stats[att] = {
                    'n_gt': 0, 'score': [], 'tp': [], 'fp': []
                    }
        self.predictions = {}


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # Store predictions for writing to file
        pred_data = []
        predictions = [pred for pred in predictions if isinstance(pred, JaadPedestrianAnnotation)]
        for pred in predictions:
            pred_data.append(pred.json_data())
        self.predictions[image_meta['image_id']] = pred_data

        # Compute metrics
        self.accumulate_attribute(predictions, image_meta,
                                    ground_truth=ground_truth)


    def accumulate_attribute(self, predictions, image_meta, *,
                             ground_truth=None):

        for att in ["is_crossing", "is_not_crossing"]:
            # Initialize ground truths
            duplicates = 0
            missed = 0
            gt_match = {}
            for gt in ground_truth:
                if (
                    gt.ignore_eval
                    or (gt.attributes[att] is None)
                    or (int(gt.attributes[att]) == 1)
                ):
                    gt_match[gt.id] = False
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[att] is not None)
                    ):
                        self.cros_stats[att]['n_gt'] += 1

            for pred in predictions:
                preds = [pred.attributes["is_not_crossing_reg"], pred.attributes["is_crossing_reg"]]
                p = argmax(preds) if att == "is_crossing" else 1 - argmax(preds)
                p = softmax(preds)[0 if att == "is_not_crossing" else 1]
                pred.attributes['score'] = pred.attributes['confidence'] * p
            # Rank predictions based on confidences
            ranked_preds = sorted(predictions, key=lambda x:x.attributes['score'], reverse=True)

            # Match predictions with closest groud truths
            for pred in ranked_preds:
                max_iou = -1.
                match = None
                for gt in ground_truth:
                    if (
                        (gt.id in gt_match)
                        and ('width' in pred.attributes)
                        and ('height' in pred.attributes)
                    ):
                        iou = compute_iou(pred.attributes['center'], pred.attributes['width'],
                                          pred.attributes['height'],
                                          gt.attributes['center'], gt.attributes['width'],
                                          gt.attributes['height'])
                    else:
                        iou = 0.
                    if (iou > 0.5) and (iou >= max_iou):

                        max_iou = iou
                        match = gt
                

                # Classify predictions as True Positives or False Positives
                if match is not None:
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[att] is not None)
                    ):
                        if not gt_match[match.id]:
                            
                            # check if True positive
                            # preds = [pred.attributes["is_not_crossing_reg"], pred.attributes["is_crossing_reg"]]
                            # p = argmax(preds) if att == "is_crossing" else 1 - argmax(preds)
                            # tp = p == gt.attributes[att] and gt.attributes[att] == 1 # redundant now that we used if (gt.id in gt_match) ?
                            tp = int(1)
                            self.cros_stats[att]['tp'].append(tp)
                            self.cros_stats[att]['fp'].append(1-tp)

                            gt_match[match.id] = True
                            score = pred.attributes['confidence'] * softmax(preds/sum(preds))[0 if att == "is_not_crossing" else 1]
                            self.cros_stats[att]['score'].append(pred.attributes['score'])
                        else:
                            duplicates+=1
                            # False positive (multiple detections)
                            score = pred.attributes['confidence'] * softmax(preds/sum(preds))[0 if att == "is_not_crossing" else 1]
                            self.cros_stats[att]['score'].append(pred.attributes['score'])
                            self.cros_stats[att]['tp'].append(0)
                            self.cros_stats[att]['fp'].append(1)

                    else:
                        # Ignore instance
                        pass
                        
                else:
                    missed+=1
                    # False positive
                    self.cros_stats[att]['score'].append(pred.attributes['confidence'])
                    self.cros_stats[att]['tp'].append(0)
                    self.cros_stats[att]['fp'].append(1)     


    def stats(self):
        text_labels = []
        stats = []

        ap = 0
        for att in ["is_crossing", "is_not_crossing"]:
            ap += (compute_ap(self.cros_stats[att]) * 0.5)
            print(self.cros_stats[att]["n_gt"], "n_gt")

        text_labels.append('hazik_instance_crossing_AP')
        stats.append(ap)


        LOG.info('will cross AP = {}'.format(ap*100))

        data = {
            'text_labels': text_labels,
            'stats': stats,
        }
        return data


    def write_predictions(self, filename, *, additional_data=None):
        with open(filename + '.pred.json', 'w') as f:
            json.dump(self.predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)



class ClassificationHazik(openpifpaf.metric.base.Base):
    """Compute classification metrics from all ground truth instances one channel classification attributes

    Args:
        attribute_metas (List[AttributeMeta]): list of meta information about
            attributes.
    """

    def __init__(self, attribute_metas: List[AttributeMeta]):
        self.attribute_metas = [am for am in attribute_metas
                                if ((am.attribute == "is_crossing_reg")
                                or (am.attribute == "is_not_crossing_reg"))]

        assert len(self.attribute_metas) == 2 # only valid classification attributes (is_crossing and is_not_crossing) are allowed
        self.cros_stats = {}
        for att in ["is_crossing", "is_not_crossing"]:
            self.cros_stats[att] = {
                'n_gt': 0, 'score': [], 'pred': [], 'true': [], 'tp': [], 'fp': []
                }
        self.predictions = {}


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # Store predictions for writing to file
        pred_data = []
        predictions = [pred for pred in predictions if isinstance(pred, JaadPedestrianAnnotation)]
        for pred in predictions:
            pred_data.append(pred.json_data())
        self.predictions[image_meta['image_id']] = pred_data

        # Compute metrics
        self.accumulate_attribute(predictions, image_meta,
                                    ground_truth=ground_truth)


    def accumulate_attribute(self, predictions, image_meta, *,
                             ground_truth=None):

        # Initialize ground truths
        ground_truth = [gt for gt in ground_truth if not gt.ignore_eval]
        
        for att in ["is_crossing", "is_not_crossing"]:
            gt_match = {}
            for gt in ground_truth:
                if (
                    gt.ignore_eval
                    or (gt.attributes[att] is None)
                    or (int(gt.attributes[att]) == 1)
                ):
                    gt_match[gt.id] = False
                    if (
                        (not gt.ignore_eval)
                        and (gt.attributes[att] is not None)
                    ):
                        self.cros_stats[att]['n_gt'] += 1
            # Match groud truths with closest predictions 
            for gt in ground_truth:

                max_iou = -1.
                match = None
                for pred in predictions:
                    if (('width' in pred.attributes)
                        and ('height' in pred.attributes)
                    ):
                        iou = compute_iou(pred.attributes['center'], pred.attributes['width'],
                                          pred.attributes['height'],
                                          gt.attributes['center'], gt.attributes['width'],
                                          gt.attributes['height'])    
                    else:
                        iou = 0.

                    if (iou > 0) and (iou >= max_iou):
                        
                        max_iou = iou
                        match = pred


                # Add predictions
                if match is not None:
                    

                    # get prediction
                    preds = [match.attributes["is_not_crossing_reg"], match.attributes["is_crossing_reg"]]
                    p = argmax(preds) if att == "is_crossing" else 1 - argmax(preds)
                    score = softmax(preds/sum(preds))[1] if att == "is_crossing" else softmax(preds/sum(preds))[0]
                    self.cros_stats[att]['score'].append(score)

                    truth = gt.attributes[att]
                    self.cros_stats[att]['pred'].append(p)
                    self.cros_stats[att]['true'].append(truth) 

                    tp = int(p == truth and truth == 1)
                    self.cros_stats[att]['tp'].append(tp)
                    self.cros_stats[att]['fp'].append(1-tp)    

                else:
                    # Default to predicting not crossing
                    p = 0 if att == "is_crossing" else 1
                    self.cros_stats[att]['score'].append(0.0)
                    self.cros_stats[att]['pred'].append(p)
                    self.cros_stats[att]['true'].append(gt.attributes[att])      
                    truth = gt.attributes[att]
                    tp = int(p == truth and truth == 1)
                    self.cros_stats[att]['tp'].append(tp)
                    self.cros_stats[att]['fp'].append(1-tp)     


    def stats(self):
        text_labels = []
        stats = []

        precision = compute_precision(self.cros_stats["is_crossing"])
        recall = compute_recall(self.cros_stats["is_crossing"])



        text_labels.append('will_cross_P')
        text_labels.append('will_cross_R')
        stats.append(precision)
        stats.append(recall)

        LOG.info('will cross P = {}'.format(precision*100))
        LOG.info('will cross R = {}'.format(recall*100))

        ap = 0
        for att in ["is_crossing", "is_not_crossing"]:
            ap += (compute_ap(self.cros_stats[att]) * 0.5)

        text_labels.append('will_cross_AP')
        stats.append(ap)

        LOG.info('will cross AP = {}'.format(ap*100))


        data = {
            'text_labels': text_labels,
            'stats': stats,
        }
        return data


    def write_predictions(self, filename, *, additional_data=None):
        with open(filename + '.pred.json', 'w') as f:
            json.dump(self.predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)

