NAME=${0%.*}

INPUT_DIR=${HOME}/yt8m_video
YT8M=${PWD}/youtube-8m

OUT_DIR=${PWD}/${NAME}
TRAIN_DIR=${PWD}/${NAME}/model

[ -d ${OUT_DIR} ] || mkdir -p ${OUT_DIR}

LANG=C; date > ${OUT_DIR}/train.log
python ${YT8M}/train.py \
       	--feature_names='mean_rgb,mean_audio' \
	--feature_sizes='1024,128' \
	--model=Cg2MoeModel \
	--moe_num_mixtures 3 \
	--base_learning_rate=0.001 \
	--num_epochs=12 \
	--train_data_pattern=${INPUT_DIR}/train*.tfrecord \
	--train_dir ${TRAIN_DIR} \
	--start_new_model >> ${OUT_DIR}/train.log 2>&1
LANG=C; date >> ${OUT_DIR}/train.log

LANG=C; date > ${OUT_DIR}/eval.log
python ${YT8M}/eval.py \
       --eval_data_pattern=${INPUT_DIR}/validate*.tfrecord \
       --moe_num_mixtures 3 \
       --train_dir ${TRAIN_DIR} \
       --run_once >> ${OUT_DIR}/eval.log 2>&1
LANG=C; date >> ${OUT_DIR}/eval.log

LANG=C; date > ${OUT_DIR}/infe.log
python ${YT8M}/inference.py \
       --input_data_pattern=${INPUT_DIR}/test*.tfrecord \
       --moe_num_mixtures 3 \
       --train_dir ${TRAIN_DIR} \
       --output_file=${OUT_DIR}/kaggle_solution.csv \
       --output_model_tgz=${OUT_DIR}/kaggle_model.tgz \
       >> ${OUT_DIR}/infe.log 2>&1
LANG=C; date >> ${OUT_DIR}/infe.log


