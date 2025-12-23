@group(0) @binding(0) var input: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn encode_ansi(@builtin(global_invocation_id) id: vec3<u32>) {
    let tex_dims = textureDimensions(input, 0);
    let idx = id.y * tex_dims.x + id.x;
    var cursor = offsets[idx];
    output[cursor / 4] = 0;
}