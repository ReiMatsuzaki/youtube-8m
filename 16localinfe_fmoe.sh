BUCKET_NAME=gs://reim2zk_us
JOB_NAME=yt8m_16train_fmoe_$(date +%Y%m%d_%H%M%S)
TRAIN_DIR=$BUCKET_NAME/yt8m_16_fmoe
#gcloud --verbosity=debug ml-engine local train \
#--package-path=youtube-8m --module-name=youtube-8m.inference \
#-- --input_data_pattern='gs://youtube8m-ml-us-east1/2/frame/test/test0001.tfrecord' \
#--frame_features=True \
#--input_model_tgz="out_16_fmoe/model_16_fmoe.tgz" \
#--output_file="out_16_fmoe/16_fmoe.csv"

NAME=16_fmoe
YT8M=youtube-8m
OUT_DIR=out_$NAME
INPUT_DIR=frame
LOGFILE=${OUT_DIR}/infe.log
LANG=C; date > $LOGFILE
python ${YT8M}/inference.py \
       --input_data_pattern=${INPUT_DIR}/test0000.tfrecord \
       --train_dir=$TRAIN_DIR \
       #--input_model_tgz=$OUT_DIR/model_$NAME.tgz \
       --output_file=${OUT_DIR}/solution_$NAME.csv
       >> $LOGFILE 2>&1
LANG=C; date >> $LOGFILE


