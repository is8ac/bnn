#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Prms {
  uint[1][16][3][3][1] data;
}
prms;

layout(set = 0, binding = 1) buffer ObjHead { uint[10][1] data; }
head;

layout(set = 0, binding = 2) buffer InputPatch {
  uint[3][3][1] data[];
}
input_patch;

layout(set = 0, binding = 3) buffer Label { uint data[]; }
label;

layout(set = 0, binding = 4) buffer Embedding { uint[1] data[]; }
embedding;

layout(set = 0, binding = 5) buffer Objective { uint data[]; }
objective;

layout(push_constant) uniform PushConstantData {
  uint embedding_word_index;
  uint embedding_bit_index;
  uint full_apply;
}
pc;

void main() {
  const uint onebits = 4294967295;
  const uint threshold = (32 * 1 * (3 * 3)) / 2;
  uint x;
  uint y;
  uint b;
  uint o;
  uint e;
  uint i;
  uint target;
  uint center_bits_sum;
  uint normal_bits_sum;
  uint fliped_bits_sum;
  uint max_obj;
  uint new_obj;
  bool is_good;

  if (pc.full_apply == 1) {
    for (e = 0; e < 1; e += 1) {
      target = 0;
      for (b = 0; b < 16; b += 1) {
        center_bits_sum = 0;
        normal_bits_sum = 0;
        fliped_bits_sum = 0;
        for (y = 0; y < 3; y += 1) {
          for (i = 0; i < 1; i += 1) {
            center_bits_sum +=
                bitCount(prms.data[e][b][1][y][i] ^
                         input_patch.data[gl_GlobalInvocationID.x][1][y][i]);
            normal_bits_sum +=
                bitCount(prms.data[e][b][0][y][i] ^
                         input_patch.data[gl_GlobalInvocationID.x][0][y][i]) +
                bitCount(prms.data[e][b][2][y][i] ^
                         input_patch.data[gl_GlobalInvocationID.x][2][y][i]);

            fliped_bits_sum +=
                bitCount(prms.data[e][b][0][y][i] ^
                         input_patch.data[gl_GlobalInvocationID.x][2][y][i]) +
                bitCount(prms.data[e][b][2][y][i] ^
                         input_patch.data[gl_GlobalInvocationID.x][0][y][i]);
          }
        }
        target |= uint(normal_bits_sum + center_bits_sum > threshold) << b;
        target |= uint(fliped_bits_sum + center_bits_sum > threshold)
                  << (16 + b);
      }
      embedding.data[gl_GlobalInvocationID.x][e] = target;
    }
  } else if (pc.full_apply == 0) {
    center_bits_sum = 0;
    normal_bits_sum = 0;
    fliped_bits_sum = 0;
    for (y = 0; y < 3; y += 1) {
      for (i = 0; i < 1; i += 1) {
        center_bits_sum +=
            bitCount(prms.data[pc.embedding_word_index][pc.embedding_bit_index][1][y][i] ^
                     input_patch.data[gl_GlobalInvocationID.x][1][y][i]);

        normal_bits_sum +=
            bitCount(prms.data[pc.embedding_word_index][pc.embedding_bit_index]
                              [0][y][i] ^
                     input_patch.data[gl_GlobalInvocationID.x][0][y][i]) +
            bitCount(prms.data[pc.embedding_word_index][pc.embedding_bit_index]
                              [2][y][i] ^
                     input_patch.data[gl_GlobalInvocationID.x][2][y][i]);

        fliped_bits_sum +=
            bitCount(prms.data[pc.embedding_word_index][pc.embedding_bit_index]
                              [0][y][i] ^
                     input_patch.data[gl_GlobalInvocationID.x][2][y][i]) +
            bitCount(prms.data[pc.embedding_word_index][pc.embedding_bit_index]
                              [2][y][i] ^
                     input_patch.data[gl_GlobalInvocationID.x][0][y][i]);
      }
    }
    target = embedding.data[gl_GlobalInvocationID.x][pc.embedding_word_index];

    target &= (onebits ^ (uint(1) << pc.embedding_bit_index));
    target &= (onebits ^ (uint(1) << (16 + pc.embedding_bit_index)));

    target |= uint(normal_bits_sum + center_bits_sum > threshold)
              << pc.embedding_bit_index;
    target |= uint(fliped_bits_sum + center_bits_sum > threshold)
              << (16 + pc.embedding_bit_index);

    embedding.data[gl_GlobalInvocationID.x][pc.embedding_word_index] = target;
  }

  // now we can start on the objective head
  is_good = true;
  max_obj = 0;
  for (e = 0; e < 1; e += 1) {
    max_obj += bitCount(head.data[label.data[gl_GlobalInvocationID.x]][e] ^
                        embedding.data[gl_GlobalInvocationID.x][e]);
  }
  for (o = 0; o < 10; o += 1) {
    new_obj = 0;
    for (e = 0; e < 1; e += 1) {
      new_obj += bitCount(head.data[o][e] ^
                          embedding.data[gl_GlobalInvocationID.x][e]);
    }
    is_good =
        ((new_obj < max_obj) || (o == label.data[gl_GlobalInvocationID.x])) &&
        is_good;
  }
  objective.data[gl_GlobalInvocationID.x] = uint(is_good);
}
