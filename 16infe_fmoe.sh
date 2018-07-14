BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_16infe_fmoe_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_16_fmoe

# copy model file and
#mkdir -p out_16_fmoe/model
#gsutil cp gs://reim2zk_us/yt8m_16_fmoe/inference_model* out_16_fmoe/model
#gsutil cp gs://reim2zk_us/yt8m_16_fmoe/model_flags.json out_16_fmoe/model/
#tar -zcvf out_16_fmoe/model.tgz out_16_fmoe/model

gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--runtime-version 1.4 \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml-us-east1/2/frame/test/test*.tfrecord' \
--frame_features=True \
--train_dir=$TRAIN_DIR \       
--input_model_tgz="16_fmoe/model.tgz" \
--output_file=$TRAIN_DIR/16_fmoe.csv



