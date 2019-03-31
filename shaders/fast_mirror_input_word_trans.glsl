#version 450

struct CacheExampleMirrored {
  // uint[2] input_word;
  uint[2] input_partial_sums;
  // uint[2] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { CacheExampleMirrored cache[]; }
cache_data;

layout(set = 0, binding = 1) buffer OldWords { uint[2] data[]; }
old_words;

layout(set = 0, binding = 2) buffer NewWords { uint[2] data[]; }
new_words;

layout(push_constant) uniform PushConstantData {
  uint new_weights_word;
  uint old_weights_word;
}
pc;

void main() {
  cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[0] =
      cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[0] +
      bitCount(pc.old_weights_word ^
               old_words.data[gl_GlobalInvocationID.x][0]) -
      bitCount(pc.new_weights_word ^
               old_words.data[gl_GlobalInvocationID.x][0]);

  cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[1] =
      cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums[1] +
      bitCount(pc.old_weights_word ^
               old_words.data[gl_GlobalInvocationID.x][1]) -
      bitCount(pc.new_weights_word ^
               old_words.data[gl_GlobalInvocationID.x][1]);
}
