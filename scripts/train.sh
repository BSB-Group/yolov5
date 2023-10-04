HOMEDIR=/home/kevinserrano
MODEL=yolov5n6
${HOMEDIR}/miniconda3/envs/AI/bin/python ${HOMEDIR}/GitHub/yolov5-extended/train.py \
    --img 1280 \
    --batch 32 \
    --epochs 50 \
    --data /home/jigglypuff/GitHub/yolov5-extended/datasets/RGB_25_17-04-2023/dataset.yaml \
    --weights yolos/yolov5n6_RGB_D2304-v0_SC.pt \
    --name ${MODEL}_RGB_D2304-v0_9C \
    --hyp ${HOMEDIR}/GitHub/yolov5-extended/data/hyps/hyp.sea-ai.yaml \
#    --workers 4 \
#    --device 0
#    --single-cls