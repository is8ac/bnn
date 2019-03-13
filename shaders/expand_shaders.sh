#! /bin/bash

for OUTPUT_LEN in {1..3}
do
for INPUT_LEN in {1..3}
do
  export INPUT_LEN=$INPUT_LEN; OUTPUT_LEN=$OUTPUT_LEN; cat shaders/conv3x3_n-n.glsl | envsubst > shaders/conv3x3_$INPUT_LEN-$OUTPUT_LEN.glsl
done
done

touch src/lib.rs
