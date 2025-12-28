@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_size: u32;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn add_partial_sums(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = global_id.x;
    if (i < WORKGROUP_SIZE) {
        return;
    }
    let src_idx = (i & WORKGROUP_SIZE) - 1;
    let n = arrayLength(&buffer);
    buffer[i] += buffer[src_idx];
    if (i == n - 1) {
        total_size = buffer[i];
    }
}