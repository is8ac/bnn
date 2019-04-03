#version 450

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint[2] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Data {
  CacheExampleMirrored cache[];
}
cache_data;

layout(set = 0, binding = 1) writeonly buffer Objective { uint data[]; }
objective_sums;

layout(push_constant) uniform PushConstantData {
  uint[10][2] head;
  uint batch_size;
}
pc;

void main() {
  uint c;
  uint i;
  uint sum = 0;
  uint index;
  uint true_class;
  for (i = 0; i < pc.batch_size; i += 1) {
    index = gl_GlobalInvocationID.x * pc.batch_size + i;
    true_class = cache_data.cache[index].true_class;

    uint[2] new_embedding = cache_data.cache[index].embedding;

    uint max_act = 0;
    uint true_act = 0;
    uint act;

    for (c = 0; c < 10; c++) {
      act = bitCount(new_embedding[0] ^ pc.head[c][0]) +
            bitCount(new_embedding[1] ^ pc.head[c][1]);
      max_act = max(max_act, ((c != true_class ? act : 0)));
      true_act += (c == true_class) ? act : 0;
    }
    sum += uint(true_act > max_act);
    //sum += 1;
  }
  objective_sums.data[gl_GlobalInvocationID.x] = sum;
}
