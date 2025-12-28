@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;
@group(0) @binding(1) var<storage, read> stride: u32;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> s_in: array<u32, WORKGROUP_SIZE>;
var<workgroup> s_out: array<u32, WORKGROUP_SIZE>;

fn prefix_sum_from_offset(offset: u32,
                          local_id: u32) {
    let size = arrayLength(&buffer);
    let buffer_idx = offset + local_id * stride;

    if (buffer_idx < size) {
        s_in[local_id] = buffer[buffer_idx];
    } else {
        s_in[local_id] = 0u;
    }

    for (var shift = 0u; shift < 8u; shift += 2u) {
        var stride = 1u << shift;
        workgroupBarrier();
        if (local_id >= stride) {
            s_out[local_id] = s_in[local_id] + s_in[local_id - stride];
        } else {
            s_out[local_id] = s_in[local_id];
        }
        stride = 1u << (shift + 1);
        workgroupBarrier();
        if (local_id >= stride) {
            s_in[local_id] = s_out[local_id] + s_out[local_id - stride];
        } else {
            s_in[local_id] = s_out[local_id];
        }
    }

    workgroupBarrier();
    if (buffer_idx < size) {
        buffer[buffer_idx] = s_in[local_id];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum(@builtin(num_workgroups) num_workgroups: vec3<u32>,
              @builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&buffer);

    for (var offset = stride - 1u; offset < size; offset += num_workgroups.x * WORKGROUP_SIZE * stride) {
        prefix_sum_from_offset(offset, local_id.x);
    }
}