#! /bin/bash

for EMBEDDING_LEN in {1..4}
do
  export EMBEDDING_LEN=$EMBEDDING_LEN
  cat shaders/fast_mirror_cache_replace_input.glsl | envsubst > shaders/fast_mirror_cache_replace_input_e${EMBEDDING_LEN}.glsl
  cat shaders/fast_mirror_embedding_bit_clean.glsl | envsubst > shaders/fast_mirror_embedding_bit_clean_e${EMBEDDING_LEN}.glsl
  cat shaders/fast_mirror_input_word_trans.glsl | envsubst > shaders/fast_mirror_input_word_trans_e${EMBEDDING_LEN}.glsl
  cat shaders/fast_mirror_head_update.glsl | envsubst > shaders/fast_mirror_head_update_e${EMBEDDING_LEN}.glsl
  #cat shaders/fast_mirror_update.glsl | envsubst > shaders/fast_mirror_update_e${EMBEDDING_LEN}.glsl
  cat shaders/fast_mirror_input_word_update.glsl | envsubst > shaders/fast_mirror_input_word_update_e${EMBEDDING_LEN}.glsl
done

touch src/lib.rs
