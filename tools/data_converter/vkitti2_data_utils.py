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
                'trackID':[], 'alpha':[], 'dimensions':[], 'location_y':[], 'rotation_y':[], 'location_w':[], 'rotation_w':[], 'truncated':[], 'occulated':[], 'isMoving':[],'num_points_in_gt':[]
            }}

            # meta data
            idx = int(key[:-2])
            anno['image']['image_idx'] = idx
            anno['image']['image_path'] =  '{}/{}/frames/rgb/Camera_0/rgb_{:05d}.jpg'.format(scene_path, event_path, idx % 10000)
            anno['image']['image_shape'] = [375, 1242]
            
            anno['calib']['P0'] = [
                [725.0087,  0.      ,  620.5,  0.],
                [0.      ,  725.0087,  187. ,  0.],
                [0.      ,  0.      ,  1.   ,  0.],
                [0.      ,  0.      ,  0.   ,  1.]]

            anno['annos']['name'] = ['Car'] * len(value)
            for column in value:
                anno['annos']['trackID'].append(column[0][2])
                anno['annos']['alpha'].append(column[0][3]) 
                anno['annos']['dimensions'].append(column[0][4:6])
                anno['annos']['location_y'].append(column[0][7:9])
                anno['annos']['rotation_y'].append(column[0][10:12])
                anno['annos']['location_w'].append(column[0][13:15])
                anno['annos']['rotation_w'].append(column[0][16:18])
                anno['annos']['truncated'].append(column[1][-3])
                anno['annos']['occulated'].append(column[1][-2])
                anno['annos']['isMoving'].append(column[1][-1])
                anno['annos']['num_points_in_gt'].append(300)
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
