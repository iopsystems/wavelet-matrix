// Translated to Rust from C code provided by Fabian Geisen:
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Note: We can implelement 4d codes by performing two 2d interleavings.
// Note: This encodes 2d as yxyxyxyx and 3d as zyxzyxzyxzyx.
// as Tropf notes in his 2021 bigmin-litmax note -- but he puts x first,
// whereas we have y first, so ours are **transposed** compared to the image below:
//
// Bitwise interleaving the 2D data (x,y) effects that the entire point field is alternatingly
// devided up in x and y direction.
//
// The first division is in x direchtion and splits it into a low x-section and a high x-section
//
// These sections are divided in y direchtion; the sections are all split into a low y-section
// and a high y-section
//
// These sections are divided in x direchtion; the sections are all split into a low x-section
// and a high x-section
//
// etc etc
//
//          ---> x
// |                       |                     |                   |
// |                       |                     |                   |
// v                       |                     |                   |
// y       -  -  -  -  -   |  -  -  - -  -  -    |  -  -  -  -  -    |  -  -  -  -  -
//                         |                     |                   |
//                         |                     |                   |
//                         |                     |                   |
//                                               |
//          -  -  -  -  -  -  -  -  - -  -  -    |    -  -  -  -  -  -  -  -  -  -  -
//                                               |
//                         |                     |                   |
//                         |                     |                   |
//                         |                     |                   |
//          -  -  -  -  -  |   -  -    -  -  -   |   -  -  -  -  -   |  -  -  -  -  -
//                         |                     |                   |
//                         |                     |                   |
//                         |                     |                   |

pub const fn encode2(x: u32, y: u32) -> u32 {
    (part_1_by_1(y) << 1) + part_1_by_1(x)
}

pub const fn encode3(x: u32, y: u32, z: u32) -> u32 {
    (part_1_by_2(z) << 2) + (part_1_by_2(y) << 1) + part_1_by_2(x)
}

pub const fn decode2x(code: u32) -> u32 {
    compact_1_by_1(code)
}

pub const fn decode2y(code: u32) -> u32 {
    compact_1_by_1(code >> 1)
}

pub const fn decode3x(code: u32) -> u32 {
    compact_1_by_2(code)
}

pub const fn decode3y(code: u32) -> u32 {
    compact_1_by_2(code >> 1)
}

pub const fn decode3z(code: u32) -> u32 {
    compact_1_by_2(code >> 2)
}

// "Insert" a 0 bit after each of the 16 low bits of x
const fn part_1_by_1(x: u32) -> u32 {
    let mut x = x;
    x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x
}

// "Insert" two 0 bits after each of the 10 low bits of x
const fn part_1_by_2(x: u32) -> u32 {
    let mut x = x;
    x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

// Inverse of part_1_by_1 - "delete" all odd-indexed bits
const fn compact_1_by_1(x: u32) -> u32 {
    let mut x = x;
    x &= 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x
}

// Inverse of part_1_by_2 - "delete" all bits not at positions divisible by 3
const fn compact_1_by_2(x: u32) -> u32 {
    let mut x = x;
    x &= 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x
}
