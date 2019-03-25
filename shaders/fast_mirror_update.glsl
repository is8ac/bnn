#version 450

struct CacheExampleMirrored {
  uint[2] input_word;
  uint[2] input_partial_sums;
  uint embedding_word;
  uint[10] head_partial_sums;
  uint true_class;
} fast_cache;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { CacheExampleMirrored cache[]; }
input_data;

layout(set = 0, binding = 1) buffer Objective { uint data[]; }
objective_sums;

// layout(set = 0, binding = 2) buffer Params {
//  uint embedding_bit_index;
//  uint threshold;
//  uint weights_word;
//  uint[10] head_words;
//}
// pc;

layout(push_constant) uniform PushConstantData {
  uint embedding_bit_index;
  uint threshold;
  uint weights_word;
  uint[10] head_words;
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

  for (i = 0; i < pc.batch_size; i += 1) {
    index = gl_GlobalInvocationID.x * pc.batch_size + i;
    true_class = input_data.cache[index].true_class;
    uint normal = uint(
        (bitCount(pc.weights_word ^
                  input_data.cache[index].input_word[0]) +
         input_data.cache[index].input_partial_sums[0]) >
        pc.threshold);

    uint fliped = uint(
        (bitCount(pc.weights_word ^
                  input_data.cache[index].input_word[1]) +
         input_data.cache[index].input_partial_sums[1]) >
        pc.threshold);

    uint new_embedding =
        input_data.cache[index].embedding_word &
            (onebits ^ (uint(1) << (0 + pc.embedding_bit_index))) &
            (onebits ^ (uint(1) << (16 + pc.embedding_bit_index))) |
        (normal << (0 + pc.embedding_bit_index)) |
        (fliped << (16 + pc.embedding_bit_index));

    uint max_obj =
        bitCount(pc.head_words[true_class] ^ new_embedding) +
        input_data.cache[index].head_partial_sums[true_class];
    bool is_good = true;
    for (c = 0; c < 10; c++) {
      is_good =
          ((c == true_class) ||
           ((bitCount(pc.head_words[c] ^ new_embedding) +
             input_data.cache[index].head_partial_sums[c]) <
            max_obj)) &&
          is_good;
    }
    sum += uint(is_good);
  }

  objective_sums.data[gl_GlobalInvocationID.x] = sum;
}
