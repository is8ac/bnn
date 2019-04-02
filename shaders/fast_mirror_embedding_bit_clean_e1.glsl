#version 450

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint[1] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { CacheExampleMirrored cache[]; }
cache_data;

layout(push_constant) uniform PushConstantData {
  uint embedding_bit_index;
  uint embedding_word_index;
  uint threshold;
  uint weights_word;
}
pc;

void main() {
  const uint onebits = 4294967295;

  uint normal =
      uint((bitCount(pc.weights_word ^
                     cache_data.cache[gl_GlobalInvocationID.x].input_word[0]) +
            cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[0]) >
           pc.threshold);

  uint fliped =
      uint((bitCount(pc.weights_word ^
                     cache_data.cache[gl_GlobalInvocationID.x].input_word[1]) +
            cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[1]) >
           pc.threshold);

  cache_data.cache[gl_GlobalInvocationID.x].embedding[pc.embedding_word_index] =
      cache_data.cache[gl_GlobalInvocationID.x].embedding[pc.embedding_word_index] &
          (onebits ^ (uint(1) << (0 + pc.embedding_bit_index))) &
          (onebits ^ (uint(1) << (16 + pc.embedding_bit_index))) |
      (normal << (0 + pc.embedding_bit_index)) |
      (fliped << (16 + pc.embedding_bit_index));
}
