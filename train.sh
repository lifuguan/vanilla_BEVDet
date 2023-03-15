export PYTHONPATH=$PWD:$PYTHONPATH
nohup bash ./tools/dist_train.sh configs/smoke/smoke-vkitti2-mono3d.py 8 > work_dirs/train.log 2>&1 &


# nohup bash ./tools/dist_train.sh configs/smoke/smoke-kitti-mono3d.py 8 > work_dirs/train.log 2>&1 &

# python tools/misc/browse_dataset.py configs/_base_/datasets/vkitti2-mono3d.py --task mono-det --output-dir work_dirs/vkitti2

# python tools/test.py configs/smoke/smoke-vkitti2-mono3d.py work_dirs/smoke-vkitti2-mono3d/epoch_72.pth --eval 'mAP' --eval-options 'show=False' 'out_dir=work_dirs/vkitti2'