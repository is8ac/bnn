#version 450

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint[2] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 8, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Data {
  CacheExampleMirrored cache[];
}
cache_data;

layout(set = 0, binding = 1) writeonly buffer Objective { uint[32] data[]; }
objective_sums;

layout(set = 0, binding = 2) readonly buffer ObjHead {
  uint[10][2] data;
}
obj_head;

layout(push_constant) uniform PushConstantData {
  uint batch_size;
  uint embedding_word_index;
  uint mut_class;
}
pc;

void main() {
  uint c;
  uint i;
  uint e;
  uint sum = 0;
  uint index;
  uint true_class;

  uint mut_head_word = obj_head.data[pc.mut_class][pc.embedding_word_index] ^
                       (uint(1) << gl_GlobalInvocationID.y);

  for (i = 0; i < pc.batch_size; i += 1) {
    index = gl_GlobalInvocationID.x * pc.batch_size + i;
    true_class = cache_data.cache[index].true_class;

    uint[2] new_embedding = cache_data.cache[index].embedding;

    uint max_act = 0;
    uint true_act = 0;
    uint act;

    for (c = 0; c < 10; c++) {
      act = 0;
      for (e = 0; e < 2; e += 1) {
        act += bitCount(new_embedding[e] ^
                        (((c == pc.mut_class) && (e == pc.embedding_word_index))
                             ? mut_head_word
                             : obj_head.data[c][e]));
      }
      max_act = max(max_act, ((c != true_class ? act : 0)));
      true_act += (c == true_class) ? act : 0;
    }
    sum += uint(true_act > max_act);
    //sum = mut_head_word;
  }
  objective_sums.data[gl_GlobalInvocationID.x][gl_GlobalInvocationID.y] = sum;
}
