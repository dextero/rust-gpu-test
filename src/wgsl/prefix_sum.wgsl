@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;
@group(0) @binding(1) var<storage, read> input_stride: u32;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> s_in: array<u32, WORKGROUP_SIZE>;
var<workgroup> s_out: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let i = local_id.x;
    let input_size = arrayLength(&buffer);
    let input_offset = input_stride - 1;
    let n = (input_size - input_offset) / input_stride;
    let buffer_idx = (input_offset + global_id.x * input_stride) % WORKGROUP_SIZE;

    if (i < n) {
        s_in[i] = buffer[buffer_idx];
    } else {
        s_in[i] = 0u;
    }

    for (var shift = 0u; shift < 8u; shift += 2u) {
        var stride = 1u << shift;
        workgroupBarrier();
        if (i >= stride) {
            s_out[i] = s_in[i] + s_in[i - stride];
        } else {
            s_out[i] = s_in[i];
        }
        stride = 1u << (shift + 1);
        workgroupBarrier();
        if (i >= stride) {
            s_in[i] = s_out[i] + s_out[i - stride];
        } else {
            s_in[i] = s_out[i];
        }
    }

    workgroupBarrier();
    if (i < n) {
        buffer[buffer_idx] = s_in[i];
    }
}