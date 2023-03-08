export PYTHONPATH=$PWD:$PYTHONPATH
nohup bash ./tools/dist_train.sh configs/smoke/smoke-vkitti2-mono3d.py 8 > work_dirs/train.log 2>&1 &