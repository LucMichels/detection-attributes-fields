import logging
import os
import sys
from typing import Callable

import openpifpaf
from PIL import Image
import torch.utils.data

from .attribute import JaadType
from . import transforms


LOG = logging.getLogger(__name__)


class JaadDataset(torch.utils.data.Dataset):
    """Dataset JAAD <http://data.nvision2.eecs.yorku.ca/JAAD_dataset/>.

    Args:
        root_dir (str): Root directory of dataset.
        split (str: 'train', 'val', 'test'): Split of dataset.
        subset (str: 'default', 'all_videos', 'high_visibility'): Set of
            videos to use.
        truncate (bool): On evaluation only consider frames before crossing
        slice (int, int): On evaluation only consider a slice of frames. First argument is where to slice as the distance in frames where the pedestrian crosses \
                           (eg: "30": from 30 frames before he crosses, "-30": from 30 frames after he crosses ) \
                           Second argument is the size in frames of the slice (takes the x previous frames')
        invert (int): Invert label of crossing pedestrians x frames before they cross (eg: "30": pedestrians is considered crossing 30 frames before he actually does
        preprocess (Callable): A function/transform that takes in the
            image and targets and transforms them.
    """

    def __init__(self,
                 root_dir: str,
                 split: str,
                 subset: str,
                 truncate: bool,
                 slice: (int, int),
                 invert: int,
                 *,
                 preprocess: Callable = None):
        super().__init__()
        sys.path.append(root_dir)
        from jaad_data import JAAD

        jaad = JAAD(data_path=root_dir)

        self.root_dir = root_dir
        if subset not in {'default', 'all_videos', 'high_visibility'}:
            raise ValueError('unknown subset {}'.format(subset))
        self.subset = subset
        if split in {'train', 'val', 'test'}:
            list_videos = jaad._get_video_ids_split(split, subset=self.subset)
        elif split == 'trainval':
            list_videos = (
                jaad._get_video_ids_split('train', subset=self.subset)
                + jaad._get_video_ids_split('val', subset=self.subset)
            )
        elif split == 'hazik_train':
            print("hazik_train")
            list_videos = self._get_split_from_range(range(1, 301))
            for s in list_videos:
                print(s)
            print("hazik_test")
            for s in self._get_split_from_range(range(301, 347)):
                print(s)
            sys.stdout.flush()
            1/0
        else:
            raise ValueError('unknown split {}'.format(split))
        self.split = split
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        self.db = jaad.generate_database()
        self.idx_to_ids = []
        for vid_id in list_videos:
            for img_id in range(self.db[vid_id]["num_frames"]):
                self.idx_to_ids.append({
                    'video_name': vid_id,
                    'image_name': '{:05d}.png'.format(img_id),
                    'frame_id': img_id,
                })

        self.truncate = truncate
        self.slice = slice
        self.invert = invert

        LOG.info('JAAD {0} {1} images: {2}'.format(self.subset, self.split,
                                                   len(self.idx_to_ids)))


    def __getitem__(self, index):
        ids = self.idx_to_ids[index]
        local_file_path = os.path.join(self.root_dir, 'images',
                                       ids['video_name'], ids['image_name'])
        with open(local_file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        # Annotations
        anns = []
        for ped_id in self.db[ids['video_name']]['ped_annotations']:
            if ids['frame_id'] not in (self.db[ids['video_name']]
                                              ['ped_annotations']
                                              [ped_id]
                                              ['frames']):
                continue # ped not present in frame
            seq_id = (self.db[ids['video_name']]
                             ['ped_annotations']
                             [ped_id]
                             ['frames']).index(ids['frame_id'])

            frames = self.db[ids['video_name']]['ped_annotations'][ped_id]['frames']
            # for i in range(1, len(frames)):
            #     diff = frames[i] - frames[i-1] 
            #     if diff > 1:
            #         print()
            #         print(f"Faulty frames: frame i-1 -> {frames[i-1]}, frame i -> {frames[i]}, frame diff -> {diff}, ped_id -> {ped_id}")
            #         sys.stdout.flush()
            #         1/0

            ped = {}
            ped['object_type'] = JaadType.PEDESTRIAN
            ped['id'] = ped_id
            ped_anns = self.db[ids['video_name']]['ped_annotations'][ped_id]
            # General
            ped['confidence'] = 1
            ped['box'] = [ # x, y, w, h
                ped_anns['bbox'][seq_id][0],
                ped_anns['bbox'][seq_id][1],
                ped_anns['bbox'][seq_id][2]-ped_anns['bbox'][seq_id][0],
                ped_anns['bbox'][seq_id][3]-ped_anns['bbox'][seq_id][1],
            ]
            ped['center'] = [ped['box'][0]+.5*ped['box'][2],
                             ped['box'][1]+.5*ped['box'][3]]
            ped['width'] = ped['box'][2]
            ped['height'] = ped['box'][3]
            ped['occlusion'] = ped_anns['occlusion'][seq_id] #0: no occlusion, 1: partial occlusion (>25%), 2: full occlusion (>75%)
            ped['with_behavior'] = True if ped_id[-1]=='b' else False
            ped['ignore_eval'] = True if ped_id[-1]=='p' else False
            # Crossing
            if 'cross' in ped_anns['behavior']:
                crossing_behavior = ped_anns['behavior']['cross']
                crossing_behavior = [max(0,cb) for cb in crossing_behavior] # replace -1 by 0
                ped['is_crossing'] = crossing_behavior[seq_id] # 0: 'not-crossing', 1: 'crossing'
                ped['will_cross'] = 1 if any(crossing_behavior[seq_id:]) else 0 # 0: 'not-crossing', 1: 'crossing'
            else:
                ped['will_cross'] = 0
                ped['is_crossing'] = 0
            if ped['will_cross'] == 1:
                cross_t = next(t for t in range(len(crossing_behavior))
                               if crossing_behavior[t]==1) # start crossing
                ped['frames_to_crossing'] = frames[cross_t] - frames[seq_id]
                ped['time_to_crossing'] = (frames[cross_t] - frames[seq_id]) / 30. # conversion to seconds at 30fps
            else:
                ped['frames_to_crossing'] = None
                ped['time_to_crossing'] = None
            # Behavior
            for tag in ['hand_gesture', 'look', 'nod', 'reaction']:
                ped[tag] = (
                    int(ped_anns['behavior'][tag][seq_id])
                    if tag in ped_anns['behavior'] else None
                )
            if (ped['hand_gesture'] is not None) and (ped['hand_gesture'] > 0):
                ped['hand_gesture'] = 1 # merge all reaction types
            ped['walk'] = ( # different name for action -> walk attribute
                int(ped_anns['behavior']['action'][seq_id])
                if 'action' in ped_anns['behavior'] else None
            )
            # Attributes
            for tag in ['age', 'gender', 'group_size', 'motion_direction']:
                ped[tag] = (
                    int(ped_anns['attributes'][tag])
                    if tag in ped_anns['attributes'] else None
                )
            if ped['age'] is not None: # merge child/young
                ped['age'] -= 1
                if ped['age'] < 0:
                    ped['age'] = 0
            if ped['gender'] is not None: # remove n/a
                ped['gender'] -= 1
                if ped['gender'] < 0:
                    ped['gender'] = None
            if ped['motion_direction'] is not None: # remove n/a
                ped['motion_direction'] -= 1
                if ped['motion_direction'] < 0:
                    ped['motion_direction'] = None
            if ped['group_size'] is not None: # limit at 4 or more
                ped['group_size'] -= 1
                if ped['group_size'] > 3:
                    ped['group_size'] = 3
            # Appearance
            if ('frames' in ped_anns['appearance']
                    and ids['frame_id'] in ped_anns['appearance']['frames']):
                app_seq_id = ped_anns['appearance']['frames'].index(ids['frame_id'])
            else:
                app_seq_id = None
            for tag in ['baby', 'backpack', 'bag_elbow', 'bag_hand',
                        'bag_left_side', 'bag_right_side', 'bag_shoulder',
                        'bicycle_motorcycle', 'cap', 'clothes_below_knee',
                        'clothes_lower_dark', 'clothes_lower_light',
                        'clothes_upper_light', 'clothes_upper_dark', 'hood',
                        'object', 'phone', 'pose_back', 'pose_front',
                        'pose_left', 'pose_right', 'stroller_cart',
                        'sunglasses', 'umbrella']:
                ped[tag] = (
                    int(ped_anns['appearance'][tag][app_seq_id])
                    if (tag in ped_anns['appearance']
                        and app_seq_id is not None)
                    else None
                )

            # Filtering
            if self.truncate and ped['will_cross']:
                if ped['frames_to_crossing'] < 0:
                    ped['ignore_eval'] = True

            if self.slice[1] > 0 and ped['will_cross']:
                slice_from, slice_size = self.slice
                if not (ped['frames_to_crossing'] - slice_from >= 0 and ped['frames_to_crossing'] - slice_from < slice_size):
                    ped['ignore_eval'] = True

            # Pre-processing invertion
            if self.invert > 0 and ped['will_cross'] and ped['frames_to_crossing'] >= 0:
                if ped['frames_to_crossing'] < self.invert:
                    ped["is_crossing"] = 1

            # Not crossing attribute
            ped["is_not_crossing"] = 1 - ped["is_crossing"]

            # Hazik regression attributess
            ped["is_not_crossing_reg"] =  ped["is_not_crossing"]*1.0
            ped["is_crossing_reg"] = ped["is_crossing"]*1.0


            # Add pedestrian
            anns.append(ped)

        meta = {
            'dataset': 'jaad',
            'dataset_index': index,
            'video_name': ids['video_name'],
            'image_name': ids['image_name'],
            'frame_id': ids['frame_id'],
            'image_id': ids['video_name'] + '/' + ids['image_name'],
            'local_file_path': local_file_path,
        }

        # Preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        LOG.debug(meta)

        return image, anns, meta


    def __len__(self):
        return len(self.idx_to_ids)

    def _get_split_from_range(self, id_range):
        split_ids = []
        for video_id in id_range:
            vid_name = f"video_{video_id:4d}"
            split_ids.append(vid_name)
        return split_ids


