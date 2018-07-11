NAME=${0%.*}
INPUT_DIR=${HOME}/yt8m/video
YT8M=${PWD}/youtube-8m

OUT_DIR=${PWD}/${NAME}
TRAIN_DIR=${PWD}/${NAME}/model

gcloud --verbosity=debug ml-engine local train  \
--package-path=youtube-8m --module-name=youtube-8m.train \
-- \
--train_data_pattern=${INPUT_DIR}/train*.tfrecord \
--model=LogisticModel \
--train_dir=${TRAIN_DIR}

#[ -d ${OUT_DIR} ] || mkdir -p ${OUT_DIR}
#LANG=C; date > ${OUT_DIR}/train.log
#python ${YT8M}/train.py \
#       	--feature_names='mean_rgb,mean_audio' \
#	--feature_sizes='1024,128' \
#	--model=LogisticModel \
#	--train_data_pattern=${INPUT_DIR}/train*.tfrecord \
#	--train_dir ${TRAIN_DIR} \
#	--start_new_model >> ${OUT_DIR}/train.log 2>&1
#LANG=C; date >> ${OUT_DIR}/train.log
