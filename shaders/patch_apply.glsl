#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Prms { uint[32][3][3] data; }
prms;

layout(set = 0, binding = 1) buffer ObjHead { uint[10] data; }
head;

layout(set = 0, binding = 2) buffer InputPatch { uint[3][3] data[]; }
input_patch;

layout(set = 0, binding = 3) buffer Label { uint data[]; }
label;

layout(set = 0, binding = 4) buffer Embedding { uint data[]; }
embedding;

layout(set = 0, binding = 5) buffer Objective { uint data[]; }
objective;

layout(push_constant) uniform PushConstantData {
  uint output_index;
  uint full_apply;
}
pc;

void main() {
  const uint onebits = 4294967295;
  const uint threshold = (32 * 9) / 2;
  uint x;
  uint y;
  uint b;
  uint o;
  uint target;
  uint sum_bits;
  uint max;

  if (pc.full_apply == 1) {
    target = 0;
    for (b = 0; b < 32; b += 1) {
      sum_bits = 0;
      for (x = 0; x < 3; x += 1) {
        for (y = 0; y < 3; y += 1) {
          sum_bits += bitCount(prms.data[b][x][y] ^
                               input_patch.data[gl_GlobalInvocationID.x][x][y]);
        }
      }
      target |= uint(sum_bits > threshold) << b;
    }
    embedding.data[gl_GlobalInvocationID.x] = target;
  } else {
    sum_bits = 0;
    for (x = 0; x < 3; x += 1) {
      for (y = 0; y < 3; y += 1) {
        sum_bits += bitCount(prms.data[pc.output_index][x][y] ^
                             input_patch.data[gl_GlobalInvocationID.x][x][y]);
      }
    }
    embedding.data[gl_GlobalInvocationID.x] &=
        (onebits ^ (uint(1) << pc.output_index));
    embedding.data[gl_GlobalInvocationID.x] |= uint(sum_bits > threshold)
                                               << pc.output_index;
  }

  // now we can start on the objective head
  max = bitCount(head.data[label.data[gl_GlobalInvocationID.x]] ^
                 embedding.data[gl_GlobalInvocationID.x]);
  for (o = 0; o < 10; o += 1) {
    if (o != label.data[gl_GlobalInvocationID.x]) {
      if (bitCount(head.data[o] ^ embedding.data[gl_GlobalInvocationID.x]) >=
          max) {
        objective.data[gl_GlobalInvocationID.x] = 0;
        return;
      }
    }
  }
  objective.data[gl_GlobalInvocationID.x] = 1;
}
