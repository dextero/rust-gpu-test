@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_size: u32;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn add_partial_sums(@builtin(num_workgroups) num_workgroups: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>,
                    @builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let elems = arrayLength(&buffer);
    let elems_per_workgroup = (elems + num_workgroups.x - 1u) / num_workgroups.x;
    let begin = workgroup_id.x * elems_per_workgroup;
    let end = min(begin + elems_per_workgroup, elems);

    var i: u32;
    for (i = begin + local_id.x; i < end; i += WORKGROUP_SIZE) {
        if ((i + 1) % WORKGROUP_SIZE) == 0 {
            continue;
        }
        let src_idx = (i & ~(WORKGROUP_SIZE - 1u)) - 1u;
        buffer[i] += buffer[src_idx];
    }

    if (i == elems - 1) {
        total_size = buffer[i];
    }
}