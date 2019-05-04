DATA=zpark

python test_DeLS-3D.py \
       --data ${DATA} \
       --pose_cnn ./models/${DATA}/pose_cnn-0000 \
       --pose_rnn ./models/${DATA}/pose_rnn-0000 \
       --seg_cnn ./model/${DATA}/seg_cnn-0000

# python train update later
