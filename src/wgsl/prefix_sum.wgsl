@group(0) @binding(0) var<storage, read_write> sizes_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_size: u32;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> s_in: array<u32, WORKGROUP_SIZE>;
var<workgroup> s_out: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let i = local_id.x;
    let n = arrayLength(&sizes_offsets);

    // This shader implements a parallel prefix sum for inputs that fit within a single workgroup.
    // It assumes n <= WORKGROUP_SIZE.

    // Load data into shared memory, padding with 0.
    if (i < n) {
        s_in[i] = sizes_offsets[i];
    } else {
        s_in[i] = 0u;
    }
    workgroupBarrier();

    // Inclusive scan using Hillis-Steele algorithm with two shared memory buffers to avoid race conditions.
    // With WORKGROUP_SIZE = 256, this requires log2(256) = 8 iterations.
    for (var d: u32 = 0; d < 8; d = d + 1) {
        let stride = 1u << d;
        if (d % 2 == 0) { // Read from s_in, write to s_out
            if (i >= stride) {
                s_out[i] = s_in[i] + s_in[i - stride];
            } else {
                s_out[i] = s_in[i];
            }
        } else { // Read from s_out, write to s_in
            if (i >= stride) {
                s_in[i] = s_out[i] + s_out[i - stride];
            } else {
                s_in[i] = s_out[i];
            }
        }
        workgroupBarrier();
    }

    // After 8 iterations, the result of the inclusive scan is in s_in.
    
    // The total sum is the last element of the inclusive scan of the original data.
    if (i == 0u) {
        if (n > 0u) {
            total_size = s_in[n - 1u];
        } else {
            total_size = 0u;
        }
    }

    // Convert the inclusive scan to an exclusive scan and write back to global memory.
    if (i < n) {
        if (i == 0u) {
            sizes_offsets[i] = 0u;
        } else {
            sizes_offsets[i] = s_in[i - 1u];
        }
    }
}
