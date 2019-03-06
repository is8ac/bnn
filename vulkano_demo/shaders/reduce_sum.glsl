#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputData {
    uint data[];
} input_data;

layout(set = 0, binding = 1) buffer OutputSum {
    uint data[];
} output_sum;

layout(push_constant) uniform PushConstantData {
    uint batch_size;
} pc;

void main() {
    uint i;

    uint sum = 0;
    uint base = gl_GlobalInvocationID.x * pc.batch_size;

    for (i=0; i<pc.batch_size; i+=1) {
        sum += input_data.data[base + i];
    }
    output_sum.data[gl_GlobalInvocationID.x] = sum;
}
