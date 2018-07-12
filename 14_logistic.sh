BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- \
--train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/train00?0.tfrecord' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/yt8m_train_video_level_logistic_model

#BUCKET_NAME=gs://reim2zk_us
#JOB_NAME=ytml_${0%.*}
#gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
#       --package-path=youtube-8m \
#       --module-name=youtube-8m.train \
#       --staging-bucket=$BUCKET_NAME \
#       --region=us-east1 \
#       --config=youtube-8m/cloudml-gpu.yaml \
#       -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/train*.tfrecord' \
#       --model=LogisticModel \
#       --train_dir=$BUCKET_NAME/$JOB_NAME


