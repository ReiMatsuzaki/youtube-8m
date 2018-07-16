# copy model file and archive it

BUCKET_NAME=gs://reim2zk_us
TRAIN_DIR=$BUCKET_NAME/yt8m_16_fmoe

out_dir=out_16_fmoe
model_dir=${out_dir}/16_model
model_tgz=${model_dir}.tgz

mkdir -p $model_dir
gsutil cp $TRAIN_DIR/inference_model* $model_dir
gsutil cp $TRAIN_DIR/model_flags.json $model_dir/
tar -zcvf $model_tgz $model_dir


