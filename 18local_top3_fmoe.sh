# calculation success.
gcloud --verbosity=debug ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train \
-- --train_data_pattern=${HOME}/calc/2018/youtube-8m/frame/train*.tfrecord \
--frame_features=True \
--model=FrameLevelTopkMoeModel \
--num_topk=3 \
--feature_names='rgb,audio' \
--feature_sizes='1024,128' \
--train_dir=out_18localtrain_topk_fmoe \
--start_new_model


