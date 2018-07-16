# copy model file and archive it

BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_17train_fcg2moe_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_17_fcg2moe

out_dir=out_17_fcg2moe
model_dir=${out_dir}/17_model
model_tgz=${model_dir}.tgz
mkdir -p $model_dir
gsutil cp $TRAIN_DIR/inference_model* $model_dir
gsutil cp $TRAIN_DIR/model_flags.json $model_dir/
tar -zcvf $model_tgz $model_dir
