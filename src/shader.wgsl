struct Pixel { r: u32, g: u32, b: u32, a: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> sizes_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

const WIDTH: u32 = 64u;

// Helper to calculate digits in a 0-255 byte
fn get_digits_len(v: u32) -> u32 {
    if (v >= 100u) { return 3u; }
    if (v >= 10u) { return 2u; }
    return 1u;
}

// --- PASS 1: Calculate Sizes ---
@compute @workgroup_size(64)
fn calc_sizes(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
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

// --- PASS 2: Prefix Sum (Simplified for minimal example) ---
// Note: Real prefix sums use a "Reduce-then-Scan" pattern. 
// For a minimal example, we'll assume a small enough buffer to scan in one workgroup.
@compute @workgroup_size(1)
fn prefix_sum() {
    var sum = 0u;
    for (var i = 0u; i < arrayLength(&sizes_offsets); i++) {
        sizes_offsets[i] = sum;
        sum += sizes_offsets[i];
    }
}

// --- PASS 3: Write String ---
fn write_byte(offset: u32, char: u32) {
    // Atomic or simple write? Since offsets are unique, we can write directly.
    // Note: We use a u32 array as a u8 buffer for convenience in this example.
    output[offset] = char;
}

// In a real implementation, you'd write a helper to convert int to ASCII
// and write bytes to the 'output' buffer. 
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * WIDTH + id.x;
    var cursor = sizes_offsets[idx];
    
    // Logic to write "\x1b[38;2;..." based on input[idx]...
    // (Actual byte-by-byte writing omitted for brevity, but uses write_byte)
}