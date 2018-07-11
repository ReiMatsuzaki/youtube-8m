NAME=${0%.*}

INPUT_DIR=${HOME}/yt8m_video
YT8M=${PWD}/youtube-8m

OUT_DIR=${PWD}/${NAME}
TRAIN_DIR=${PWD}/${NAME}/model

[ -d ${OUT_DIR} ] || mkdir -p ${OUT_DIR}

LANG=C; date > ${OUT_DIR}/eval.log
python ${YT8M}/eval.py \
       --eval_data_pattern=${INPUT_DIR}/validate*.tfrecord \
       --moe_num_mixtures 3 \
       --train_dir ${PWD}/02_cg2moe3/kaggle_model \
       --run_once >> ${OUT_DIR}/eval.log >&1
LANG=C; date >> ${OUT_DIR}/eval.log


