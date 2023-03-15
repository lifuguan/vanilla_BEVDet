# Copyright (c) lifuguan. All rights reserved.
#%%
from concurrent import futures as futures
import concurrent.futures
from pathlib import Path
from concurrent import futures as futures
import concurrent.futures
import numpy as np
import mmcv

#%%


def get_vkitti2_image_info(path, training=True):
    annos =[]
    event_path = '15-deg-left' if training == True else 'clone'

    def process_scene(scene_path, event_path, path):
        label_path1 = Path(path) / scene_path / event_path / "pose.txt"
        label_path2 = Path(path) / scene_path / event_path / "bbox.txt"

        contents1 = np.loadtxt(label_path1, delimiter=' ', skiprows=1)
        contents2 = []
        with open(label_path2, 'r') as f:   
            f.readline()
            for line in f:
                contents2.append(line.split())

        frame = np.array([str(10000* int(scene_path[-2:]) + x[0]) for x in contents1]) 
        num_frame = np.unique(frame)
        dic = { frame_:[] for frame_ in num_frame}

        for content1, content2 in zip(contents1, contents2):
            if content1[1] == 0:
                dic[str(10000* int(scene_path[-2:]) + content1[0])].append([content1, content2]) 

        annos = []
        for key, value in mmcv.track_iter_progress(dic.items()):
            anno = {"image":{}, "calib":{}, "annos":{
                'trackID':np.array([]), 'alpha':np.array([]), 'dimensions':np.empty((0,3)), 'location':np.empty((0,3)), 'rotation_y':np.empty((0,3)), 'location_w':np.empty((0,3)), 'rotation_w':np.empty((0,3)), 'truncated':np.array([]), 'occluded':np.array([]), 'isMoving':np.array([]),'num_points_in_gt':np.array([]),'bbox':np.empty((0,4))}}

            # meta data
            idx = int(key[:-2])
            anno['image']['image_idx'] = idx
            anno['image']['image_path'] =  '{}/{}/frames/rgb/Camera_0/rgb_{:05d}.jpg'.format(scene_path, event_path, idx % 10000)
            anno['image']['image_shape'] = [375, 1242]
            
            anno['calib']['P0'] = np.array(
                [[725.0087,  0.     ,  620.5,  0.],
                [0.      ,  725.0087,  187. ,  0.],
                [0.      ,  0.      ,  1.   ,  0.],
                [0.      ,  0.      ,  0.   ,  1.]])

            anno['annos']['name'] = np.array(['Car'] * len(value))
            for column in value:
                anno['annos']['trackID'] = np.append(anno['annos']['trackID'], column[0][2])
                anno['annos']['alpha'] = np.append(anno['annos']['alpha'], column[0][3]) 

                # vkitti2:width, height, length;   kitti: height, width, length
                anno['annos']['dimensions'] = np.append(anno['annos']['dimensions'], [[column[0][5],column[0][4],column[0][6]]], axis=0)
                anno['annos']['location_w']= np.append(anno['annos']['location_w'], [column[0][7:10]], axis=0)
                anno['annos']['rotation_w']= np.append(anno['annos']['rotation_w'], column[0][10])  # 只使用y轴
                anno['annos']['location']= np.append(anno['annos']['location'], [column[0][13:16]], axis=0)
                # anno['annos']['rotation_y']= np.append(anno['annos']['rotation_y'], column[0][16])  # 只使用y轴
                anno['annos']['rotation_y']= np.append(anno['annos']['rotation_y'], column[0][16] - 1.57)  # 只使用y轴

                # vkitti2:left, right, top, bottom;   kitti: left, top, right, bottom
                anno['annos']['bbox'] = np.append(anno['annos']['bbox'] , [np.float_([column[1][3],column[1][5],column[1][4],column[1][6]])], axis=0)
                anno['annos']['truncated'] = np.append(anno['annos']['truncated'] , float(column[1][-3]))

                # vkitti2: 0: fully occluded, 1: fully visible; kitti: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
                occluded = 3 - int(float(column[1][-2]) / 0.25)
                anno['annos']['occluded'] = np.append(anno['annos']['occluded']  , occluded)
                anno['annos']['isMoving'] = np.append(anno['annos']['isMoving']  , column[1][-1])
                anno['annos']['num_points_in_gt'] = np.append(anno['annos']['num_points_in_gt'], 300)
            annos.append(anno)
        
        return annos

    scene_paths = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交场景处理任务
        futures = [executor.submit(process_scene, scene_path, event_path, path) for scene_path in scene_paths]
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            annos.extend(future.result())

    print("convert finished")
    return annos


# res = get_vkitti2_image_info("/home/hao/lihao/data/vkitti2", training=True)

# %%


# def process_scene(scene_path, event_path, path):
#     label_path1 = Path(path) / scene_path / event_path / "pose.txt"
#     label_path2 = Path(path) / scene_path / event_path / "bbox.txt"

#     contents1 = np.loadtxt(label_path1, delimiter=' ', skiprows=1)
#     contents2 = []
#     with open(label_path2, 'r') as f:   
#         f.readline()
#         for line in f:
#             contents2.append(line.split())

#     frame = np.array([str(10000* int(scene_path[-2:]) + x[0]) for x in contents1]) 
#     num_frame = np.unique(frame)
#     dic = { frame_:[] for frame_ in num_frame}

#     for content1, content2 in zip(contents1, contents2):
#         if content1[1] == 0:
#             dic[str(10000* int(scene_path[-2:]) + content1[0])].append([content1, content2]) 

#     annos = []
#     for key, value in mmcv.track_iter_progress(dic.items()):
#     # for key, value in dic.items():
#         anno = {"image":{}, "annos":{
#             'trackID':[], 'alpha':[], 'dimensions':[], 'location_y':[], 'rotation_y':[], 'location_w':[], 'rotation_w':[], 'truncated':[], 'occulated':[], 'isMoving':[],'num_points_in_gt':[]
#         }}
#         idx = int(key[:-2])
#         anno['image']['image_idx'] = idx
#         anno['image']['image_path'] =  '{}/{}/frames/rgb/Camera_0/rgb_{:05d}.jpg'.format(scene_path, event_path, idx % 10000)
#         anno['image']['image_shape'] = [375, 1242]
        
#         anno['annos']['name'] = ['Car'] * len(value)
#         for column in value:
#             anno['annos']['trackID'].append(column[0][2])
#             anno['annos']['alpha'].append(column[0][3]) 
#             anno['annos']['dimensions'].append(column[0][4:6])
#             anno['annos']['location_y'].append(column[0][7:9])
#             anno['annos']['rotation_y'].append(column[0][10:12])
#             anno['annos']['location_w'].append(column[0][13:15])
#             anno['annos']['rotation_w'].append(column[0][16:18])
#             anno['annos']['truncated'].append(column[1][-3])
#             anno['annos']['occulated'].append(column[1][-2])
#             anno['annos']['isMoving'].append(column[1][-1])
#             anno['annos']['num_points_in_gt'].append(300)
#         annos.append(anno)
    
#     return annos

# if __name__ == '__main__':
#     scene_paths = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
#     event_path = '15-deg-left'
#     path = '/home/hao/lihao/data/vkitti2'

#     annos = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # 提交场景处理任务
#         futures = [executor.submit(process_scene, scene_path, event_path, path) for scene_path in scene_paths]
#         # 等待所有任务完成
#         for future in concurrent.futures.as_completed(futures):
#             annos.extend(future.result())
    
#     # 在此处使用处理后的结果
#     print("annos")

