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

    for (var i = local_id.x; i < elems_per_workgroup; i += WORKGROUP_SIZE) {
        let src_idx = (i & WORKGROUP_SIZE) - 1u;
        buffer[i] += buffer[src_idx];
    }

    if (workgroup_id.x == num_workgroups.x - 1 && local_id.x == WORKGROUP_SIZE - 1) {
        total_size = buffer[elems - 1];
    }
}