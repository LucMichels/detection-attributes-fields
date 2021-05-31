import copy

import numpy as np
from openpifpaf.utils import mask_valid_area
import torch
import sys

from .encoder import AnnotationRescaler, AttributeGenerator


class BoxAnnotationRescaler(AnnotationRescaler):
    """AnnotationRescaler for objects defined with bounding boxes."""

    def objects(self, anns):
        objs = [copy.deepcopy(ann) for ann in anns
                if ann['object_type'] is self.object_type]
        for obj in objs:
            obj['box'] /= self.stride
            obj['center'] /= self.stride
            obj['width'] /= self.stride
            obj['height'] /= self.stride
        return objs


class BoxAttributeGenerator(AttributeGenerator):
    """AttributeGenerator for objects defined with bounding boxes."""
    
    rescaler_class = BoxAnnotationRescaler


    def generate_encoding(self, objects, width_height, valid_area):
        self.init_fields(width_height)
        self.fill(objects)
        encodings = self.fields(valid_area)
        return encodings


    def init_fields(self, width_height):
        init_value = np.nan if self.config.meta.only_on_instance else 0.
        assert self.config.meta.n_channels > 0
        n_targets = (1 if self.config.meta.is_classification
                     else self.config.meta.n_channels)
        self.targets = np.full(
            (n_targets, width_height[1], width_height[0]),
            init_value,
            dtype=np.float32,
        )
        self.previous_distances = np.full((width_height[1], width_height[0]),
                                          np.inf, dtype=np.float32)
        self.previous_bottoms = np.full((width_height[1], width_height[0]),
                                        -1., dtype=np.float32)


    def fill(self, objects):
        for obj in objects:
            self.fill_object(obj)


    def fill_object(self, obj):
        x_start = int(np.round(obj['box'][0]))
        x_end = int(np.round(obj['box'][0] + obj['box'][2]) + 1)
        y_start = int(np.round(obj['box'][1]))
        y_end = int(np.round(obj['box'][1] + obj['box'][3]) + 1)
        mask_size = [x_end - x_start, y_end - y_start]

        target_mask = self.target_mask(obj, mask_size)

        v_center = np.stack((
            np.linspace(
                obj['center'][0] - np.round(obj['box'][0]),
                obj['center'][0] - np.round(obj['box'][0] + obj['box'][2]),
                mask_size[0],
            ).reshape(1,-1).repeat(mask_size[1], axis=0),
            np.linspace(
                obj['center'][1] - np.round(obj['box'][1]),
                obj['center'][1] - np.round(obj['box'][1] + obj['box'][3]),
                mask_size[1],
            ).reshape(-1,1).repeat(mask_size[0], axis=1),
        ), axis=0)
        d_center = np.linalg.norm(v_center, ord=2, axis=0)
        t = self.targets[:, y_start:y_end, x_start:x_end]
        pd = self.previous_distances[y_start:y_end, x_start:x_end]
        pb = self.previous_bottoms[y_start:y_end, x_start:x_end]

        if (t.shape[1] <= 0) or (t.shape[2] <= 0):
            return

        # No learning on heavily occluded or ignored instances
        if (
            (obj['occlusion'] > self.config.occlusion_level)
            or obj['ignore_eval']
        ):
            if not self.config.meta.only_on_instance:
                t[t==0.] = np.nan
            return

        valid_mask = (
            (pd > d_center)
            | ((pd == d_center) & (pb < obj['box'][1]+obj['box'][3]))
        )
        t[
            np.expand_dims(valid_mask, axis=0).repeat(t.shape[0], axis=0)
        ] = target_mask[
            np.expand_dims(valid_mask, axis=0).repeat(target_mask.shape[0],
                                                      axis=0)
        ]

        pd[valid_mask] = d_center[valid_mask]
        pb[valid_mask] = obj['box'][1] + obj['box'][3]


    def target_mask(self, obj, mask_size):
        val = obj[self.config.meta.attribute]

        if self.config.meta.is_scalar:
            if val is None:
                val = np.nan
            target = np.full((1, mask_size[1], mask_size[0]),
                             val, dtype=np.float32)
            if self.config.meta.mean is not None:
                target -= self.config.meta.mean
            if self.config.meta.std is not None:
                target /= self.config.meta.std
        else: # vectorial attribute
            if val is None:
                val = [np.nan, np.nan]
            target = np.stack((
                np.linspace(
                    val[0] - np.round(obj['box'][0]),
                    val[0] - np.round(obj['box'][0] + obj['box'][2]),
                    mask_size[0],
                ).reshape(1,-1).repeat(mask_size[1], axis=0),
                np.linspace(
                    val[1] - np.round(obj['box'][1]),
                    val[1] - np.round(obj['box'][1] + obj['box'][3]),
                    mask_size[1],
                ).reshape(-1,1).repeat(mask_size[0], axis=1),
            ), axis=0)
            if self.config.meta.mean is not None:
                target[0,:,:] -= self.config.meta.mean[0]
                target[1,:,:] -= self.config.meta.mean[1]
            if self.config.meta.std is not None:
                target[0,:,:] /= self.config.meta.std[0]
                target[1,:,:] /= self.config.meta.std[1]

        return target


    def fields(self, valid_area):
        mask_valid_area(self.targets, valid_area, fill_value=np.nan)
        return torch.from_numpy(self.targets)

class BoxGaussianAttributeGenerator(AttributeGenerator):
    """AttributeGenerator for objects defined with bounding boxes."""
    
    rescaler_class = BoxAnnotationRescaler


    def generate_encoding(self, objects, width_height, valid_area):
        self.init_fields(width_height)
        self.fill(objects)
        encodings = self.fields(valid_area)
        return encodings


    def init_fields(self, width_height):
        init_value = np.nan if self.config.meta.only_on_instance else 0.
        init_value = 0. if self.config.meta.group == 'hazik' else init_value
        assert self.config.meta.n_channels > 0
        n_targets = (1 if self.config.meta.is_classification
                     else self.config.meta.n_channels)
        self.targets = np.full(
            (n_targets, width_height[1], width_height[0]),
            init_value,
            dtype=np.float32,
        )
        self.previous_distances = np.full((width_height[1], width_height[0]),
                                          np.inf, dtype=np.float32)
        self.previous_bottoms = np.full((width_height[1], width_height[0]),
                                        -1., dtype=np.float32)


    def fill(self, objects):
        for obj in objects:
            self.fill_object(obj)


    def fill_object(self, obj):
        x_start = int(np.round(obj['box'][0]))
        x_end = int(np.round(obj['box'][0] + obj['box'][2]) + 1)
        y_start = int(np.round(obj['box'][1]))
        y_end = int(np.round(obj['box'][1] + obj['box'][3]) + 1)
        mask_size = [x_end - x_start, y_end - y_start]

        target_mask = self.target_mask(obj, mask_size)

        v_center = np.stack((
            np.linspace(
                obj['center'][0] - np.round(obj['box'][0]),
                obj['center'][0] - np.round(obj['box'][0] + obj['box'][2]),
                mask_size[0],
            ).reshape(1,-1).repeat(mask_size[1], axis=0),
            np.linspace(
                obj['center'][1] - np.round(obj['box'][1]),
                obj['center'][1] - np.round(obj['box'][1] + obj['box'][3]),
                mask_size[1],
            ).reshape(-1,1).repeat(mask_size[0], axis=1),
        ), axis=0)
        d_center = np.linalg.norm(v_center, ord=2, axis=0)
        t = self.targets[:, y_start:y_end, x_start:x_end]
        pd = self.previous_distances[y_start:y_end, x_start:x_end]
        pb = self.previous_bottoms[y_start:y_end, x_start:x_end]

        if (t.shape[1] <= 0) or (t.shape[2] <= 0):
            return

        # No learning on heavily occluded or ignored instances
        if (
            (obj['occlusion'] > self.config.occlusion_level)
            or obj['ignore_eval']
        ):
            if not self.config.meta.only_on_instance:
                t[t==0.] = np.nan
            return

        valid_mask = (
            (pd > d_center)
            | ((pd == d_center) & (pb < obj['box'][1]+obj['box'][3]))
        )


        # add the gaussian scaling
        if self.config.meta.group == "hazik":
            # generate the distribution centered at this box
            w = x_end-x_start
            h = y_end-y_start
            x0, y0, sigma_x, sigma_y = x_start+float(w)/2, y_start+float(h)/2, float(w)/4, float(h)/4
            # activity map for current person
            y, x = np.arange(self.targets.shape[1]), np.arange(self.targets.shape[2])    
            gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
            gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
            g  = np.outer(gy, gx)
            self.targets = np.amax([self.targets, np.expand_dims(g*obj[self.config.meta.attribute], axis=0).astype(np.float32)], axis=0).astype(np.float32)

            cx, cy = x_start+w//2+1, y_start+h//2+1

            print(cx, cy, w, h)
            print("gaussian", g[y_start:y_start+h+1, x_start:x_start+w+1])
            print("center",g[cy:cy+1,cx:cx+1])
            print("targets", self.targets[:,y_start:y_start+h+1, x_start:x_start+w+1])
            print("center",self.targets[:,cy:cy+1,cx:cx+1])
            print("pred", obj[self.config.meta.attribute])
            sys.stdout.flush()
            1/0
        else:
            t[
                np.expand_dims(valid_mask, axis=0).repeat(t.shape[0], axis=0)
            ] = target_mask[
                np.expand_dims(valid_mask, axis=0).repeat(target_mask.shape[0],
                                                          axis=0)
            ]


        pd[valid_mask] = d_center[valid_mask]
        pb[valid_mask] = obj['box'][1] + obj['box'][3]


    def target_mask(self, obj, mask_size):
        val = obj[self.config.meta.attribute]

        if self.config.meta.is_scalar:
            if val is None:
                val = np.nan
            target = np.full((1, mask_size[1], mask_size[0]),
                             val, dtype=np.float32)
            if self.config.meta.mean is not None:
                target -= self.config.meta.mean
            if self.config.meta.std is not None:
                target /= self.config.meta.std
        else: # vectorial attribute
            if val is None:
                val = [np.nan, np.nan]
            target = np.stack((
                np.linspace(
                    val[0] - np.round(obj['box'][0]),
                    val[0] - np.round(obj['box'][0] + obj['box'][2]),
                    mask_size[0],
                ).reshape(1,-1).repeat(mask_size[1], axis=0),
                np.linspace(
                    val[1] - np.round(obj['box'][1]),
                    val[1] - np.round(obj['box'][1] + obj['box'][3]),
                    mask_size[1],
                ).reshape(-1,1).repeat(mask_size[0], axis=1),
            ), axis=0)
            if self.config.meta.mean is not None:
                target[0,:,:] -= self.config.meta.mean[0]
                target[1,:,:] -= self.config.meta.mean[1]
            if self.config.meta.std is not None:
                target[0,:,:] /= self.config.meta.std[0]
                target[1,:,:] /= self.config.meta.std[1]

        return target


    def fields(self, valid_area):
        mask_valid_area(self.targets, valid_area, fill_value=np.nan)
        return torch.from_numpy(self.targets)

