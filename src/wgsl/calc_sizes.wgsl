@group(0) @binding(0) var input: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> sizes: array<u32>;

fn get_digits_len(val: u32) -> u32 {
    var len: u32 = 1;
    if (val > 9) { len += 1; }
    if (val > 99) { len += 1; }
    return len;
}

@compute @workgroup_size(16, 16, 1)
fn calc_sizes(@builtin(global_invocation_id) id: vec3<u32>) {
    let tex_dims = textureDimensions(input, 0);
    let pos_top = vec2(id.x, id.y * 2);
    let pos_bot = vec2(id.x, id.y * 2 + 1);

    if (pos_top.x >= tex_dims.x || pos_top.y >= tex_dims.y) {
        return;
    }

    let top = textureLoad(input, pos_top, 0);

    // \x1b[38;RRR;GGG;BBBm
    //         ^^^ ^^^ ^^^                      
    //          '---'---'---------------------------------- 1-3 * 3
    // ^~~~^^^^   ^   ^   ^                                 8 fixed chars
    //                     \x1b[48;RRR;GGG;BBBm
    //                             ^^^ ^^^ ^^^                      
    //                              '---'---'-------------- 1-3 * 3
    //                     ^~~~^^^^   ^   ^   ^^~~~^~~~^~~~ 8 fixed chars
    //                                         \xe2\x96\x80
    //                                         ^~~~^~~~^~~~ 3 fixed chars
    //                                                     \n if at EOL
    var len = 11u; // 8 fixed + 3 for upper half block
    len += get_digits_len(top.r) + get_digits_len(top.g) + get_digits_len(top.b);
    // Only add the second escape if there is an odd row
    if (pos_bot.y < tex_dims.y) {
        let bot = textureLoad(input, pos_bot, 0);
        len += 8u; // 8 fixed
        len += get_digits_len(bot.r) + get_digits_len(bot.g) + get_digits_len(bot.b);
    }
    
    // Add newline if at the end of a row
    if (id.x == tex_dims.x - 1) { len += 1u; } // "\n"

    let out_idx = id.y * tex_dims.x + id.x;
    sizes[out_idx] = len;
}