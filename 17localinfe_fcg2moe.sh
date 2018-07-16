out_dir=out_17_fcg2moe
model_dir=${out_dir}/17_model
model_tgz=${model_dir}.tgz

out_dir=out_17_fcg2moe
model_dir=${out_dir}/17_model
model_tgz=${model_dir}.tgz

gcloud --verbosity=debug ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train \
-- --train_data_pattern=frame/test*.tfrecord \
--frame_features=True \
--input_model_tgz=$model_tgz \
--output_file=$out_dir/16_submission.csv


