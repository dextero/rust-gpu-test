@group(0) @binding(0) var input: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

const ANSI_ESCAPE: u32 = 0x1b;
const ASCII_LEFT_BRACKET: u32 = 0x5b;
const ASCII_ZERO: u32 = 0x30;
const ASCII_SEMICOLON: u32 = 0x3b;
const ASCII_LOWERCASE_M: u32 = 0x6d;
const UTF8_UPPER_HALF_BLOCK: array<u32, 3> = array<u32, 3>(0xe2, 0x96, 0x80);

struct Appender {
    acc: u32,
    acc_mask: u32,
    acc_shift: u32,
    dest_off: u32,
}

fn flush(appender: ptr<function, Appender>) {
    let old_val = output[appender.dest_off] & ~appender.acc_mask;
    let new_val = appender.acc & appender.acc_mask;
    output[appender.dest_off] = old_val | new_val;
    appender.acc = 0;
    appender.acc_mask = 0;
    appender.acc_shift = 0;
    appender.dest_off += 1;
}

fn append(appender: ptr<function, Appender>,
          byte: u32) {
    appender.acc |= (byte & 0xFF) << appender.acc_shift;
    appender.acc_shift += 8;
    appender.acc_mask = (appender.acc_mask << 8) | 0xFF;
    if (appender.acc_shift >= 32) {
        flush(appender);
    }
}

fn append_u8_str(appender: ptr<function, Appender>,
              value: u32) {
    let ones = value % 10;
    let tens = (value % 100) / 10;
    let hundreds = (value % 1000) / 100;
    if (hundreds > 0) {
        append(appender, ASCII_ZERO + hundreds);
    }
    if (tens > 0) {
        append(appender, ASCII_ZERO + tens);
    }
    append(appender, ASCII_ZERO + ones);
}

fn append_rgba(appender: ptr<function, Appender>,
               pixel: vec4<u32>,
               foreground: bool) {
    append(appender, ANSI_ESCAPE);
    append(appender, ASCII_LEFT_BRACKET);
    if (foreground) {
        append(appender, ASCII_ZERO + 3);
    } else {
        append(appender, ASCII_ZERO + 4);
    }
    append(appender, ASCII_ZERO + 8);
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

@compute @workgroup_size(64)
fn encode_ansi(@builtin(global_invocation_id) id: vec3<u32>) {
    let tex_dims = textureDimensions(input, 0);
    let idx = id.y * tex_dims.x + id.x;
    var cursor = offsets[idx];

    var appender = Appender();
    appender.acc = 0;
    appender.acc_mask = 0;
    appender.acc_shift = cursor % 4;
    appender.dest_off = cursor / 4;

    let pix = textureLoad(input, vec2(0, 0), 0);
    append_rgba(&appender, pix, true);
    append_rgba(&appender, pix, false);
    append(&appender, UTF8_UPPER_HALF_BLOCK[0]);
    append(&appender, UTF8_UPPER_HALF_BLOCK[1]);
    append(&appender, UTF8_UPPER_HALF_BLOCK[2]);
    flush(&appender);
}