## TODO List

- [x] Add horizon heads to yolov5 `DetectionModel` --> `HorizonModel`
- [x] Create augmentation pipeline (image + keypoints)
- [x] Create first training loop to get a feeling of results
  - 👎😥 Poor performance when training only classification heads (all other layers are frozen)
  - 👍😎 Good performance when fine-tuning backbone + classification heads
  - 😐 We might need to follow an `EnsembleModel` approach for now (no shared knowledge)
- [x] Remove unneeded layers from `HorizonModel` (`DetectionModel`'s neck and head)
- [ ] Implement metrics for assessing horizon detection
  - [ ] Take argmax of classification output --> straightforward value and confidence
  - [ ] Interpret output of classification heads as a distribution and take mean and std as value and confidence?
  - [ ] Same as before but with "peak suppression" or GMMs (describe this scenario in more detail with example pictures)
- [x] Refactor code (/horizon/transforms.py, /horizon/train.py)
- [ ] Implement `EnsembleModel` (horizon + bboxes)
- [ ] Implement "horizon-check" for datasets (check for outliers in theta and pitch)
- [ ] Implement logger (compatible with `wandb`)
- [ ] Export `EnsembleModel` to TensorRT
- [ ] Speed benchmarks for `EnsembleModel`
- [ ] Speed up training by:
  - [ ] Optimising fiftyone dataloader
  - [ ] Having a yolo-like format for the horizon annotations
- [ ] Document results and how to reproduce them

## Future work

- Multi-task learning --> train horizon + bboxes at the same time
  - How to deal with images without a horizon annotation? just because the line is not visible does not mean there is no horizon, one could argue that a human could extrapolate from other visual cues (such as patterns in the water or nearby structures)
  - How to deal with mosaic augmentation? Good for bboxes but not suitable for horizon?
  - Should we modify and extend the existing yolov5 train.py? or should we build something from "scratch"?
