MODEL=yolov5n6_trained4singleCls
/home/jigglypuff/anaconda3/envs/yolo/bin/python /home/jigglypuff/GitHub/yolov5-extended/train.py \
    --img 1280 \
    --batch 32 \
    --epochs 20 \
    --data /home/jigglypuff/GitHub/yolov5-extended/datasets/RGB_25_24-04-2023/dataset.yaml \
    --weights ${MODEL}.pt \
    --name RGB_25k_fineTuneSnglCls_modifiedTrainVal_${MODEL} \
    --hyp /home/jigglypuff/GitHub/yolov5-extended/data/hyps/hyp.sea-ai.yaml \