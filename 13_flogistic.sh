NAME=${0%.*}

INPUT_DIR=${HOME}/yt8m/frame
YT8M=${PWD}/youtube-8m

OUT_DIR=${PWD}/out_${NAME}
TRAIN_DIR=${PWD}/out_${NAME}/model

[ -d ${OUT_DIR} ] || mkdir -p ${OUT_DIR}

python ${YT8M}/train.py \
       --frame_features \
       --model=FrameLevelLogisticModel \
       --feature_names='rgb,audio' \
       --feature_sizes='1024,128' \
       --train_data_pattern=${INPUT_DIR}/train*.tfrecord \
       --train_dir ${TRAIN_DIR} \
       --start_new_model >> ${OUT_DIR}/train.log 2>&1


#LANG=C; date > ${OUT_DIR}/eval.log
#python ${YT8M}/eval.py \
#       --eval_data_pattern=${INPUT_DIR}/validate*.tfrecord \
#       --train_dir ${TRAIN_DIR} \
#       --run_once >> ${OUT_DIR}/eval.log 2>&1
#LANG=C; date >> ${OUT_DIR}/eval.log
#
#LANG=C; date > ${OUT_DIR}/infe.log
#python ${YT8M}/inference.py \
#       --input_data_pattern=${INPUT_DIR}/test*.tfrecord \
#       --moe_num_mixtures 2 \
#       --train_dir ${TRAIN_DIR} \
#       --output_file=${OUT_DIR}/kaggle_solution.csv \
#       --output_model_tgz=${OUT_DIR}/kaggle_model.tgz \
#       >> ${OUT_DIR}/infe.log 2>&1
#LANG=C; date >> ${OUT_DIR}/infe.log


