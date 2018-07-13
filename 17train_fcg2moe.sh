BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_16train_fcg2moe_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_17_fcg2moe
gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
--frame_features=True \
--model=FrameLevelCg2MoeModel \
--feature_names='rgb,audio' \
--feature_sizes='1024,128' \
--train_dir=$BUCKET_NAME/$JOB_NAME \
--start_new_model

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


