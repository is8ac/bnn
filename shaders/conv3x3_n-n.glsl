#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Prms {
  uint[${OUTPUT_LEN}][32][3][3][${INPUT_LEN}] data;
}
prms;

layout(set = 0, binding = 1) buffer ObjHead { uint[10][${OUTPUT_LEN}] data; }
head;

layout(set = 0, binding = 2) buffer InputPatch {
  uint[3][3][${INPUT_LEN}] data[];
}
input_patch;

layout(set = 0, binding = 3) buffer Label { uint data[]; }
label;

layout(set = 0, binding = 4) buffer Embedding { uint[${OUTPUT_LEN}] data[]; }
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
  const uint threshold = (32 * ${INPUT_LEN} * 9) / 2;
  uint x;
  uint y;
  uint b;
  uint o;
  uint e;
  uint i;
  uint target;
  uint sum_bits;
  uint max_obj;
  uint new_obj;
  bool is_good;

  if (pc.full_apply == 1) {
    for (e = 0; e < ${OUTPUT_LEN}; e += 1) {
      target = 0;
      for (b = 0; b < 32; b += 1) {
        sum_bits = 0;
        for (x = 0; x < 3; x += 1) {
          for (y = 0; y < 3; y += 1) {
            for (i = 0; i < ${INPUT_LEN}; i += 1) {
              sum_bits +=
                  bitCount(prms.data[e][b][x][y][i] ^
                           input_patch.data[gl_GlobalInvocationID.x][x][y][i]);
            }
          }
        }
        target |= uint(sum_bits > threshold) << b;
      }
      embedding.data[gl_GlobalInvocationID.x][e] = target;
    }
  } else if (pc.full_apply == 0) {
    sum_bits = 0;
    for (x = 0; x < 3; x += 1) {
      for (y = 0; y < 3; y += 1) {
        for (i = 0; i < ${INPUT_LEN}; i += 1) {
          sum_bits +=
              bitCount(prms.data[pc.embedding_word_index]
                                [pc.embedding_bit_index][x][y][i] ^
                       input_patch.data[gl_GlobalInvocationID.x][x][y][i]);
        }
      }
    }
    embedding.data[gl_GlobalInvocationID.x][pc.embedding_word_index] &=
        (onebits ^ (uint(1) << pc.embedding_bit_index));
    embedding.data[gl_GlobalInvocationID.x][pc.embedding_word_index] |=
        uint(sum_bits > threshold) << pc.embedding_bit_index;
  }

  // now we can start on the objective head
  is_good = true;
  max_obj = 0;
  for (e = 0; e < ${OUTPUT_LEN}; e += 1) {
    max_obj += bitCount(head.data[label.data[gl_GlobalInvocationID.x]][e] ^
                        embedding.data[gl_GlobalInvocationID.x][e]);
  }
  for (o = 0; o < 10; o += 1) {
    new_obj = 0;
    for (e = 0; e < ${OUTPUT_LEN}; e += 1) {
      new_obj += bitCount(head.data[o][e] ^
                          embedding.data[gl_GlobalInvocationID.x][e]);
    }
    is_good =
        ((new_obj < max_obj) || (o == label.data[gl_GlobalInvocationID.x])) &&
        is_good;
  }
  objective.data[gl_GlobalInvocationID.x] = uint(is_good);
}
