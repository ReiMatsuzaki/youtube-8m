# copy model file and archive it

BUCKET_NAME=gs://reim2zk_us
TRAIN_DIR=$BUCKET_NAME/yt8m_16_fmoe

OUT_DIR=out_16_fmoe
MODEL_NAME=model_16_fmoe

mkdir -p $OUT_DIR/$MODEL_NAME
gsutil cp $TRAIN_DIR/inference_model* $OUT_DIR/$MODEL_NAME
gsutil cp $TRAIN_DIR/model_flags.json $OUT_DIR/$MODEL_NAME
cd $OUT_DIR/$MODEL_NAME; tar zcvf $MODEL_NAME.tgz *
mv $MODEL_NAME.tgz ../




