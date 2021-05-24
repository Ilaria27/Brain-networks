#!/bin/bash

rm -rf ../autism


for dropout_rate in 0 0.1 0.2
do
for feat in 5 10 15
do
for c in 0.7 0.85 1
do

python "/content/drive/My Drive/competitors/GroupINN/groupinn.py" autism h_1 a_1 -t 80 --b --dropout_rate $dropout_rate --c_train $c --feature_reduction $feat

done
done
done

