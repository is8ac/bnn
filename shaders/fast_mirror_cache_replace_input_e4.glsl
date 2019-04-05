#version 450

struct NewCacheParts {
  uint[2] input_word;
  uint[2] input_partial_sums;
} cache_parts;

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint[4] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) writeonly buffer Data {
  CacheExampleMirrored cache[];
}
cache_data;

layout(set = 0, binding = 1) readonly buffer Objective { NewCacheParts data[]; }
new_cache_parts;

void main() {
  cache_data.cache[gl_GlobalInvocationID.x].input_word =
      new_cache_parts.data[gl_GlobalInvocationID.x].input_word;

  cache_data.cache[gl_GlobalInvocationID.x].input_partial_sums =
      new_cache_parts.data[gl_GlobalInvocationID.x].input_partial_sums;
}
