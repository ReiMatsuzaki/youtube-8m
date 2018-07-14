#BUCKET_NAME=gs://reim2zk_us

# copy model file and
#mkdir -p out_16_fmoe/model
#gsutil cp gs://reim2zk_us/yt8m_16_fmoe/inference_model* out_16_fmoe/model
#gsutil cp gs://reim2zk_us/yt8m_16_fmoe/model_flags.json out_16_fmoe/model/
#tar -zcvf out_16_fmoe/model.tgz out_16_fmoe/model

gcloud --verbosity=debug ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.inference \
-- --input_data_pattern='gs://youtube8m-ml-us-east1/2/frame/test/test0001.tfrecord' \
--frame_features=True \
--input_model_tgz="out_16_fmoe/model.tgz" \
--output_file="out_16_fmoe/16_fmoe.csv"



