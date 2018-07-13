# calculation success.
TRAIN_DIR=out_18localtrain_topk_fmoe
gcloud --verbosity=debug ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train \
-- --train_data_pattern=${HOME}/calc/2018/youtube-8m/frame/train*.tfrecord \
--frame_features=True \
--model=MeanStdTopkVideoModel \
--video_level_classifier=MoeModel \
--num_topk=3 \
--moe_num_mixtures 3 \
--feature_names='rgb,audio' \
--feature_sizes='1024,128' \
--train_dir=$TRAIN_DIR \
--start_new_model


