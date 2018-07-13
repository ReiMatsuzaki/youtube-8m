# success in local
BUCKET_NAME=gs://reim2zk_us
gcloud --verbosity=debug ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train \
-- --train_data_pattern=${HOME}/calc/2018/youtube-8m/frame/train*.tfrecord \
--frame_features=True \
--model=FrameLevelCg2MoeModel \
--feature_names='rgb,audio' \
--feature_sizes='1024,128' \
--train_dir=out_17localtrain_fcg2moe \
--start_new_model
