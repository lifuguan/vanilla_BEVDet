#%% 
import mmcv

pkl_file = mmcv.load("/home/hao/lihao/data/kitti/kitti_infos_val.pkl", file_format='pkl')
json_file = mmcv.load("/home/hao/lihao/data/kitti/kitti_infos_val_mono3d.coco.json", file_format='json')
print("hello")

vk2_pkl = mmcv.load("/home/hao/lihao/data/vkitti2/mono3d_infos_train.pkl", file_format='pkl')
vk2_json = mmcv.load("/home/hao/lihao/data/vkitti2/mono3d_infos_train.coco.json", file_format='json')
print("hello")