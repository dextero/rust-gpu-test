@group(0) @binding(0) var input: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

const WORKGROUP_SIZE: u32 = 256;

const ANSI_ESCAPE: u32 = 0x1b;
const ASCII_LEFT_BRACKET: u32 = 0x5b;
const ASCII_ZERO: u32 = 0x30;
const ASCII_SEMICOLON: u32 = 0x3b;
const ASCII_LOWERCASE_M: u32 = 0x6d;
const ASCII_NEWLINE: u32 = 0x0a;
const UTF8_UPPER_HALF_BLOCK: array<u32, 3> = array<u32, 3>(0xe2, 0x96, 0x80);

struct Appender {
    acc: u32,
    acc_shift: u32,
    dest_off: u32,
}

fn pad(appender: ptr<function, Appender>) {
    const padding: array<u32, 3> = array<u32, 3>(ANSI_ESCAPE, ASCII_LEFT_BRACKET, ASCII_ZERO + 3);
    let padding_needed = (4 - appender.acc_shift / 8) % 4;
    for (var i = 0u; i < padding_needed; i += 1) {
        append(appender, padding[i]);
    }
}

fn flush(appender: ptr<function, Appender>) {
    output[appender.dest_off] = appender.acc;
    appender.acc = 0;
    appender.acc_shift = 0;
    appender.dest_off += 1;
}

fn append(appender: ptr<function, Appender>,
          byte: u32) {
    appender.acc |= (byte & 0xFF) << appender.acc_shift;
    appender.acc_shift += 8;
    if (appender.acc_shift >= 32) {
        flush(appender);
    }
}

fn append_u8_str(appender: ptr<function, Appender>,
              value: u32) {
    let ones = value % 10;
    let tens = (value % 100) / 10;
    let hundreds = (value % 1000) / 100;
    if (value > 99) {
        append(appender, ASCII_ZERO + hundreds);
    }
    if (value > 9) {
        append(appender, ASCII_ZERO + tens);
    }
    append(appender, ASCII_ZERO + ones);
}

fn append_rgba(appender: ptr<function, Appender>,
               pixel: vec4<u32>,
               foreground: bool,
               skip: u32) {
    if (skip <= 0) { append(appender, ANSI_ESCAPE); }
    if (skip <= 1) { append(appender, ASCII_LEFT_BRACKET); }
    if (skip <= 2) {
        if (foreground) {
            append(appender, ASCII_ZERO + 3);
        } else {
            append(appender, ASCII_ZERO + 4);
        }
    }
    if (skip <= 3) { append(appender, ASCII_ZERO + 8); }
    append(appender, ASCII_SEMICOLON);
    append(appender, ASCII_ZERO + 2);
    append(appender, ASCII_SEMICOLON);
    append_u8_str(appender, pixel.r & 0xFF);
    append(appender, ASCII_SEMICOLON);
    append_u8_str(appender, pixel.g & 0xFF);
    append(appender, ASCII_SEMICOLON);
    append_u8_str(appender, pixel.b & 0xFF);
    append(appender, ASCII_LOWERCASE_M);
}

fn encode_single_char(char_pos: vec2<u32>) {
    let tex_dims = textureDimensions(input, 0);
    let pos_top = vec2(char_pos.x, char_pos.y * 2);
    let pos_bot = vec2(pos_top.x, pos_top.y + 1);
    if (pos_top.x >= tex_dims.x || pos_top.y >= tex_dims.y) {
        return;
    }

    let idx = char_pos.y * tex_dims.x + char_pos.x;
    var cursor = 0u;
    if idx > 0 {
        cursor = offsets[idx - 1];
    }

    var appender = Appender();
    appender.acc = 0;
    appender.acc_shift = 0;
    // Convert dest_off to multiple of u32s, rounding up. Skipped part is
    // written by previous thread.
    appender.dest_off = (cursor + 3) / 4;
    let skip = appender.dest_off * 4 - cursor;

    let pix_top = textureLoad(input, pos_top, 0);
    append_rgba(&appender, pix_top, true, skip);
    if (pos_bot.y < tex_dims.y) {
        let pix_bot = textureLoad(input, pos_bot, 0);
        append_rgba(&appender, pix_bot, false, /*skip=*/0);
    }
    append(&appender, UTF8_UPPER_HALF_BLOCK[0]);
    append(&appender, UTF8_UPPER_HALF_BLOCK[1]);
    append(&appender, UTF8_UPPER_HALF_BLOCK[2]);
    if (pos_top.x == tex_dims.x - 1) {
        append(&appender, ANSI_ESCAPE);
        append(&appender, ASCII_LEFT_BRACKET);
        append(&appender, ASCII_ZERO);
        append(&appender, ASCII_LOWERCASE_M);
        append(&appender, ASCII_NEWLINE);
    }
    pad(&appender);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn encode_ansi(@builtin(num_workgroups) num_workgroups: vec3<u32>,
               @builtin(workgroup_id) workgroup_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tex_dims = textureDimensions(input, 0);
    let total_chars = tex_dims.x * (tex_dims.y + 1) / 2;
    let chars_per_workgroup = (total_chars + num_workgroups.x - 1) / num_workgroups.x;
    let char_begin = workgroup_id.x * chars_per_workgroup;
    let char_end = min(char_begin + chars_per_workgroup, total_chars);

    for (var char_off = char_begin; char_off < char_end; char_off += WORKGROUP_SIZE) {
        let idx = char_off + local_id.x;
        let char_pos = vec2(idx % tex_dims.x, idx / tex_dims.x);
        encode_single_char(char_pos);
    }
}