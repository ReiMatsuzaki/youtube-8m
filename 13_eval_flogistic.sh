BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_13_eval_flogistic
TRAIN_DIR=$BUCKET_NAME/yt8m_13_flogistic

gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- \
--eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--frame_features \
--model=FrameLevelLogisticModel \
--train_dir=$TRAIN_DIR \
--run_once=True
