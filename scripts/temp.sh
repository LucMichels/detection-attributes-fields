# all train
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
#

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

# lambdas0
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
  --weight-decay 0 \
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

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 4 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

    --save-all ${xpdir}/images \
    --show-final-image \
    --show-final-ground-truth \

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_coco_${evalepoch} \
    --checkpoint ${evalfrom} \
    --decoder=cifcaf:0 \
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done
# detection
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
# all train
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

# warm
echo "Start training..."
mkdir -p ${xpdir}/checkpoints
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 8 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"


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
  --lr-warm-up-start-epoch 250 \
  --weight-decay 5e-4 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"
for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([2-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_coco_${evalepoch} \
    --checkpoint ${evalfrom} \
    --decoder=cifcaf:0 \
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

#multiple datasets
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
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

#pifpaf detect

mkdir -p ${xpdir}/predictions
mkdir -p ${xpdir}/images
echo "Start training..."
mkdir -p ${xpdir}/checkpoints
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 8 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
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
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

# inverted
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --jaad-invert 99999 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 99999 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done


for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_coco_${evalepoch} \
    --checkpoint ${evalfrom} \
    --decoder=cifcaf:0 \
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --jaad-invert 99999 \
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
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

# gaussian

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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --jaad-invert 99999 \
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
    --dataset=cocokp --force-complete-pose --seed-threshold=0.2 \
    --cocokp-val-annotations ${COCO_ANNOTATIONS_VAL} \
    --cocokp-val-image-dir ${COCO_IMAGE_DIR_VAL} \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch+([0-9]))
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 99999 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

validation_epoch=254
for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_0_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 0 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --decoder-s-threshold ${sthreshold} \
    --decoder-optics-min-cluster-size ${minclustersize} \
    --decoder-optics-epsilon ${epsilon} \
    --decoder-optics-cluster-threshold ${clusterthreshold} \
    --decoder-use-pifpaf-bbox \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

# regression hazik eval
validation_epoch=251
for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done


for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_0_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 0 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done
# train regression
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 0.01 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --weight-decay 1e-05 \
  --momentum 0.95 \
  --checkpoint resnet50 \
  --attribute-regression-loss l2 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --jaad-invert 99999 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

# hazik split
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
  --cocokp-orientation-invariant 0.1 \
  --cocokp-upsample=2 \
  --coco-eval-orientation-invariant 0.0 \
  --ema 1e-3 \
  --auto-tune-mtl \
  --log-interval 10 \
  --val-interval 1 \
  --val-batches 1 \
  --epochs ${epochs} \
  --batch-size 4 \
  --lr ${lr} \
  --weight-decay 0 \
  --momentum 0.95 \
  --checkpoint ${checkpoint} \
  --attribute-regression-loss l2 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes ${attributes} \
  --lambdas ${lambdas} \
  --jaad-invert 99999 \
  2>&1 | tee ${xpdir}/logs/train_log.txt
echo "Training done!"

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done



for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_0_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 0 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_4_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 120 \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_1_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 30 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_2_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 60 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

for evalfrom in $(ls -1v ${xpdir}/checkpoints/*.pt.epoch${validation_epoch})
do
  echo "Start evaluating ${evalfrom}..."
  evalepoch=${evalfrom: -3}
  srun time python3 -m openpifpaf.eval \
    --output ${xpdir}/predictions/model_jaad_invert_t_3_${evalepoch} \
    --dataset jaad \
    --jaad-root-dir /work/vita/datasets/JAAD/ \
    --jaad-subset ${jaadsubset} \
    --jaad-testing-set ${evalsplit} \
    --checkpoint ${evalfrom} \
    --batch-size 6 \
    --jaad-head-upsample 2 \
    --jaad-pedestrian-attributes ${attributes} \
    --head-consolidation=keep \
    --force-complete-pose --seed-threshold=0.2 \
    --jaad-invert 90 \
    --jaad-truncate \
    2>&1 | tee ${xpdir}/logs/eval_${evalepoch}_log.txt
  echo "Evaluation done!"
done

