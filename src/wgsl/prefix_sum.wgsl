@group(0) @binding(0) var<storage, read_write> sizes_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_size: u32;

@compute @workgroup_size(1)
fn prefix_sum() {
    var sum = 0u;
    for (var i = 0u; i < arrayLength(&sizes_offsets); i++) {
        let next = sum + sizes_offsets[i];
        sizes_offsets[i] = sum;
        sum = next;
    }
    total_size = sum;
}
