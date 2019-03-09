#! /bin/bash

for INPUT_LEN in {1..5}
do
  export INPUT_LEN=$INPUT_LEN; cat shaders/patch_apply_n_input.glsl | envsubst > shaders/patch_apply_$INPUT_LEN-1.glsl
done
