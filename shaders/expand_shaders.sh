#! /bin/bash

for OUTPUT_LEN in {1..4}
do
  export OUTPUT_LEN=$OUTPUT_LEN
  for INPUT_LEN in {1..4}
  do
    export INPUT_LEN=$INPUT_LEN
    for PATCH_SIZE in 2 3
    do
      export PATCH_SIZE=$PATCH_SIZE
      cat shaders/conv_patch_n-n.glsl | envsubst > shaders/conv${PATCH_SIZE}x${PATCH_SIZE}_$INPUT_LEN-$OUTPUT_LEN.glsl
    done
    cat shaders/mirror3x3_n-n.glsl | envsubst > shaders/mirror3x3_$INPUT_LEN-$OUTPUT_LEN.glsl
  done
done

touch src/lib.rs
