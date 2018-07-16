# copy model file and archive it

BUCKET_NAME=gs://reim2zk_us
TRAIN_DIR=$BUCKET_NAME/yt8m_17_fcg2moe

OUT_DIR=out_17_fcg2moe
MODEL_NAME=model_17_fcg2moe

mkdir -p $OUT_DIR/$MODEL_NAME
gsutil cp $TRAIN_DIR/inference_model* $OUT_DIR/$MODEL_NAME
gsutil cp $TRAIN_DIR/model_flags.json $OUT_DIR/$MODEL_NAME
cd $OUT_DIR; tar zcvf $MODEL_NAME.tgz $MODEL_NAME/


