#version 450

struct CacheExampleMirrored {
  // uint[2] input_word;
  uint[2] input_partial_sums;
  // uint[1] embedding;
  uint true_class;
} fast_cache;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputWords { uint[2] data[]; }
input_words;

layout(set = 0, binding = 1) readonly buffer Data {
  CacheExampleMirrored cache[];
}
cache_data;

layout(set = 0, binding = 2) readonly buffer Embeddings { uint[1] data[]; }
embeddings;

layout(set = 0, binding = 3) writeonly buffer Objective { uint data[]; }
objective_sums;

// layout(set = 0, binding = 2) buffer Params {
//  uint embedding_bit_index;
//  uint threshold;
//  uint weights_word;
//  uint[10] head_words;
//}
// pc;

layout(push_constant) uniform PushConstantData {
  uint[10][1] head;
  uint embedding_bit_index;
  uint embedding_word;
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
  uint batch_id = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y;
  for (i = 0; i < pc.batch_size; i += 1) {
    index = batch_id * pc.batch_size + i;
    true_class = cache_data.cache[index].true_class;
    uint normal =
        uint((bitCount(pc.weights_word ^ input_words.data[index][0]) +
              cache_data.cache[index].input_partial_sums[0]) > pc.threshold);

    uint fliped =
        uint((bitCount(pc.weights_word ^ input_words.data[index][1]) +
              cache_data.cache[index].input_partial_sums[1]) > pc.threshold);

    uint[1] new_embedding;
    new_embedding[0] =
        embeddings.data[index][0] &
            (onebits ^ (uint(1) << (0 + pc.embedding_bit_index))) &
            (onebits ^ (uint(1) << (16 + pc.embedding_bit_index))) |
        (normal << (0 + pc.embedding_bit_index)) |
        (fliped << (16 + pc.embedding_bit_index));

    uint max_act = 0;
    uint true_act = 0;
    uint act;

    for (c = 0; c < 10; c++) {
      act = bitCount(new_embedding[0] ^ pc.head[c][0]);
      max_act = max(max_act, ((c != true_class ? act : 0)));
      true_act += (c == true_class) ? act : 0;
    }
    sum += uint(true_act > max_act);
  }
  objective_sums.data[batch_id] = sum;
}
