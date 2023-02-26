# Copyright (c) lifuguan. All rights reserved.
import mmcv
from .vkitti2_data_utils import get_vkitti2_image_info

def create_kitti_info_file(data_path):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """

    kitti_infos_train = get_vkitti2_image_info(data_path, training=True)
    filename = data_path / f'mono3d_infos_train.json'
    print(f'VKitti2 info train file is saved to {filename}')
    mmcv.dump(kitti_infos_train, filename)

    filename = data_path / f'mono3d_infos_test.json'
    kitti_infos_test = get_vkitti2_image_info(data_path, training=False)
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(kitti_infos_test, filename)

