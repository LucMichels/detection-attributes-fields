

srun time python3 -m openpifpaf.train \
  --output ${xpdir}/checkpoints/model.pt \
  --dataset ${dataset} \
  --jaad-root-dir /work/vita/datasets/JAAD/ \
  --jaad-subset ${jaadsubset} \
  --jaad-training-set ${trainsplit} \
  --jaad-validation-set ${evalsplit} \
  --cocokp-train-annotations ${COCO_ANNOTATIONS_TRAIN} \
  --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
  --cocokp-train-image-dir ${COCO_IMAGE_DIR_TRAIN} \
  --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
  --cocokp-upsample=2 \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --ema 0 \
  --lr-warm-up-start-epoch -1 \
  --weight-decay 0 \
  --momentum 0 \
  --basenet resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --auto-tune-mtl \
  --dataset-weights 1.0 0.5 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_coco_${evalepoch} \
    --checkpoint ${evalfrom} \
    --decoder=cifcaf:0 \
    --head-consolidation="keep" \
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

# no lr0
srun time python3 -m openpifpaf.train \
  --output ${xpdir}/checkpoints/model.pt \
  --dataset ${dataset} \
  --jaad-root-dir /work/vita/datasets/JAAD/ \
  --jaad-subset ${jaadsubset} \
  --jaad-training-set ${trainsplit} \
  --jaad-validation-set ${evalsplit} \
  --cocokp-train-annotations ${COCO_ANNOTATIONS_TRAIN} \
  --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
  --cocokp-train-image-dir ${COCO_IMAGE_DIR_TRAIN} \
  --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
  --cocokp-upsample=2 \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --lr-warm-up-start-epoch -1 \
  --weight-decay 5e-4 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --dataset-weights 0.5 1.0 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

# swap

srun time python3 -m openpifpaf.train \
  --output ${xpdir}/checkpoints/model.pt \
  --dataset ${dataset} \
  --no-nesterov \
  --jaad-root-dir /work/vita/datasets/JAAD/ \
  --jaad-subset ${jaadsubset} \
  --jaad-training-set ${trainsplit} \
  --jaad-validation-set ${evalsplit} \
  --cocokp-train-annotations ${COCO_ANNOTATIONS_TRAIN} \
  --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
  --cocokp-train-image-dir ${COCO_IMAGE_DIR_TRAIN} \
  --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
  --cocokp-upsample=2 \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --ema 0 \
  --lr-warm-up-start-epoch -1 \
  --weight-decay 0 \
  --momentum 0 \
  --basenet resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --auto-tune-mtl \
  --dataset-weights 1.0 0.5 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

# detection
echo "Start training..."
mkdir -p ${xpdir}/checkpoints
srun time python3 -m openpifpaf.train \
  --output ${xpdir}/checkpoints/model.pt \
  --dataset ${dataset} \
  --jaad-root-dir /work/vita/datasets/JAAD/ \
  --jaad-subset ${jaadsubset} \
  --jaad-training-set ${trainsplit} \
  --jaad-validation-set ${evalsplit} \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --lr-warm-up-start-epoch -1 \
  --weight-decay 5e-4 \
  --momentum 0.95 \
  --basenet fn-resnet50 \
  --pifpaf-pretraining \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --fork-normalization-operation ${mtlgradmerge} \
  --fork-normalization-duplicates ${duplicates} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --auto-tune-mtl \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"