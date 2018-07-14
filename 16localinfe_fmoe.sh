BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_16infe_fmoe_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_16_fmoe
gcloud --verbosity=debug ml-engine \
local train \
--package-path=youtube-8m --module-name=youtube-8m.inference \
-- --input_data_pattern='gs://youtube8m-ml-us-east1/2/frame/test/test*.tfrecord' \
--frame_features=True \
--train_dir=$TRAIN_DIR \
--output_file=output_16_fmoe.csv \
--output_model_tgz=output_16_fmoe_model.tgz


