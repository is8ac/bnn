#version 450

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint embedding_word;
  uint true_class;
} fast_cache;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Data { CacheExampleMirrored cache[]; }
cache_data;

layout(set = 0, binding = 1) writeonly buffer Objective { uint data[]; }
objective_sums;

// layout(set = 0, binding = 2) buffer Params {
//  uint embedding_bit_index;
//  uint threshold;
//  uint weights_word;
//  uint[10] head_words;
//}
// pc;

layout(push_constant) uniform PushConstantData {
  uint[10] head_words;
  uint embedding_bit_index;
  uint threshold;
  uint weights_word;
  uint batch_size;
}
pc;

void main() {
  const uint onebits = 4294967295;

  uint c;
  uint i;
  uint sum = 0;
  uint index;
  uint true_class;
  uint batch_id =
      gl_GlobalInvocationID.x * gl_NumWorkGroups.y + gl_GlobalInvocationID.y;
  for (i = 0; i < pc.batch_size; i += 1) {
    index = batch_id * pc.batch_size + i;
    true_class = cache_data.cache[index].true_class;
    uint normal = uint(
        (bitCount(pc.weights_word ^ cache_data.cache[index].input_word[0]) +
         cache_data.cache[index].input_partial_sums[0]) > pc.threshold);

    uint fliped = uint(
        (bitCount(pc.weights_word ^ cache_data.cache[index].input_word[1]) +
         cache_data.cache[index].input_partial_sums[1]) > pc.threshold);

    uint new_embedding =
        cache_data.cache[index].embedding_word &
            (onebits ^ (uint(1) << (0 + pc.embedding_bit_index))) &
            (onebits ^ (uint(1) << (16 + pc.embedding_bit_index))) |
        (normal << (0 + pc.embedding_bit_index)) |
        (fliped << (16 + pc.embedding_bit_index));

    //cache_data.cache[index].embedding_word = new_embedding;

    uint max_act = 0;
    uint true_act = 0;
    uint act;

    for (c = 0; c < 10; c++) {
      act = bitCount(new_embedding ^ pc.head_words[c]);
      max_act = max(max_act, ((c != true_class ? act : 0)));
      true_act += (c == true_class) ? act : 0;
    }
    sum += uint(true_act > max_act);
  }
  objective_sums.data[batch_id] = sum;
}
