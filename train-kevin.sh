MODEL=yolov5m
/home/jigglypuff/anaconda3/envs/yolo/bin/python /home/jigglypuff/GitHub/yolov5-extended/train.py \
    --img 640 \
    --batch 32 \
    --epochs 20 \
    --data /home/jigglypuff/GitHub/yolov5-extended/datasets/RGB_137k_31-03-2023/dataset.yaml \
    --weights ${MODEL}.pt \
    --name RGB_single-cls_${MODEL} \
    --hyp /home/jigglypuff/GitHub/yolov5-extended/data/hyps/hyp.sea-ai.yaml \
    --single-cls
    
