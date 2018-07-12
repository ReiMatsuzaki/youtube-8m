BUCKET_NAME=gs://reim2zk_us
NAME=yt8m_16_fmoe
JOB_NAME=yt8m_16train_fmoe_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
--frame_features \						   
--model=FrameLevelMoeModel \
--feature_names='rgb,audio' \
--feature_sizes='1024,128' \
--train_dir=$BUCKET_NAME/$NAME \
--start_new_model

