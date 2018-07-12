BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_13eval_logistic_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_train_video_level_logistic_model

gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_dir=$TRAIN_DIR --eval_data_pattern='gs://youtube8m-ml-us-east1/2/video/validate/validate*.tfrecord' --model=LogisticModel --run_once=True
