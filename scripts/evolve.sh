MODEL=yolov5s
/home/jigglypuff/anaconda3/envs/yolo/bin/python /home/jigglypuff/GitHub/yolov5-extended/train.py \
    --img 640 \
    --batch 64 \
    --epochs 10 \
    --data /home/jigglypuff/GitHub/yolov5-extended/datasets/RGB_127k_21-03-2023/dataset.yaml \
    --weights ${MODEL}.pt \
    --name RGB_127k_evolve_${MODEL} \
    --evolve