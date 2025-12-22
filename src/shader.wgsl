struct Pixel { r: u32, g: u32, b: u32, a: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(1) @binding(0) var<storage, read_write> sizes_offsets: array<u32>;
@group(2) @binding(0) var<storage, read_write> output: array<u32>;

const WIDTH: u32 = 64;

fn get_digits_len(val: u32) -> u32 {
    var len: u32 = 1;
    if (val > 9) { len += 1; }
    if (val > 99) { len += 1; }
    return len;
}

@compute @workgroup_size(64)
fn calc_sizes(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let idx = y * WIDTH + x;
    
    let top = input[y * 2u * WIDTH + x];
    let bot = input[(y * 2u + 1u) * WIDTH + x];

    // ANSI structure: \x1b[38;2;R;G;Bm \x1b[48;2;R;G;Bm ▀
    // Base chars: "\x1b[38;2;;;m\x1b[48;2;;;m▀" = 22 fixed characters
    var len = 22u;
    len += get_digits_len(top & 0xFFu) + get_digits_len((top >> 8u) & 0xFFu) + get_digits_len((top >> 16u) & 0xFFu);
    len += get_digits_len(bot & 0xFFu) + get_digits_len((bot >> 8u) & 0xFFu) + get_digits_len((bot >> 16u) & 0xFFu);
    
    // Add newline if at the end of a row
    if (x == WIDTH - 1u) { len += 5u; } // "\x1b[0m\n"

    sizes_offsets[idx] = len;
}

@compute @workgroup_size(1)
fn prefix_sum() {
    var sum = 0u;
    for (var i = 0u; i < arrayLength(&sizes_offsets); i++) {
        sizes_offsets[i] = sum;
        sum += sizes_offsets[i];
    }
}

@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * WIDTH + id.x;
    var cursor = sizes_offsets[idx];
}