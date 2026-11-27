//! Dequantization kernels.  All offsets follow the memory layout of the
//! corresponding `ggml` C structs (see `ggml-common.h`).

use super::ggml_type::*;

/// Number of scalar elements in one block of this type.
pub const fn block_elements(kind: u32) -> usize {
    match kind {
        F32 | F64 => 1,
        F16 | BF16 => 1,
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
        Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => 256,
        _ => 1,
    }
}

/// Number of bytes in one block of this type.
pub const fn block_bytes(kind: u32) -> usize {
    match kind {
        F32 => 4,
        F64 => 8,
        F16 | BF16 => 2,
        Q4_0 => 18,  // f16 d + 16 nibble bytes
        Q4_1 => 20,  // f16 d + f16 m + 16 nibble bytes
        Q5_0 => 22,  // f16 d + u32 qh + 16 bytes
        Q5_1 => 24,  // f16 d + f16 m + u32 qh + 16 bytes
        Q8_0 => 34,  // f16 d + 32 i8
        Q8_1 => 36,  // f16 d + 32 i8 + f16 sum
        Q2_K => 84,  // 16 scales + 64 qs + f16 d + f16 dmin
        Q3_K => 110, // 32 hmask + 64 qs + 12 scales + f16 d
        Q4_K => 144, // f16 d + f16 dmin + 12 scales + 128 qs
        Q5_K => 176, // f16 d + f16 dmin + 12 scales + 32 qh + 128 qs
        Q6_K => 210, // 128 ql + 64 qh + 16 i8 scales + f16 d
        Q8_K => 292, // f32 d + 256 i8 + 16 i16 sums
        Q1_0 => 18,  // f16 d + 16 bytes of 1-bit quants
        Q2_0 => 18,  // f16 d + 16 bytes of 2-bit quants
        TQ1_0 => 54, // 48 qs + 4 qh + f16 d (ternary, 1.6875 bpw)
        TQ2_0 => 66, // 64 qs + f16 d (ternary, 2.0625 bpw)
        PQ2_0 => 34,  // f16 d + 32 bytes of 2-bit quants (Prism, group 128)
        PTQ1_0 => 28, // 24 qs + 2 qh + f16 d (Prism, group 128)
        PQ2_0M => 36, // f16 d + f16 m + 32 qs (KV only)
        _ => 0,
    }
}

/// Whether this crate can dequantize the given GGML type.
pub const fn is_supported(kind: u32) -> bool {
    block_bytes(kind) != 0
}

/// Human readable name of a GGML type.
pub fn type_name(kind: u32) -> &'static str {
    match kind {
        F32 => "F32",
        F16 => "F16",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q8_0 => "Q8_0",
        Q8_1 => "Q8_1",
        Q2_K => "Q2_K",
        Q3_K => "Q3_K",
        Q4_K => "Q4_K",
        Q5_K => "Q5_K",
        Q6_K => "Q6_K",
        Q8_K => "Q8_K",
        IQ2_XXS => "IQ2_XXS",
        IQ2_XS => "IQ2_XS",
        IQ3_XXS => "IQ3_XXS",
        IQ1_S => "IQ1_S",
        IQ4_NL => "IQ4_NL",
        IQ3_S => "IQ3_S",
        IQ2_S => "IQ2_S",
        IQ4_XS => "IQ4_XS",
        I8 => "I8",
        I16 => "I16",
        I32 => "I32",
        I64 => "I64",
        F64 => "F64",
        IQ1_M => "IQ1_M",
        BF16 => "BF16",
        TQ1_0 => "TQ1_0",
        TQ2_0 => "TQ2_0",
        Q1_0 => "Q1_0",
        Q2_0 => "Q2_0",
        PQ2_0 => "PQ2_0",
        PTQ1_0 => "PTQ1_0",
        PQ2_0M => "PQ2_0M",
        _ => "UNKNOWN",
    }
}

#[inline]
fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn f32le(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32le(b, o))
}

/// IEEE 754 binary16 -> binary32 conversion, without external crates.
#[inline]
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half: normalize into a regular f32.
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Byte order of the extra `s` field in a Q8_1 block.
///
/// Older ggml files store `{ d, qs[32], s }`, newer ones `{ d, s, qs[32] }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q8Layout {
    Old,
    New,
}

/// Decode `elements` scalars of type `kind` from `src` into `dst`.
///
/// Returns `false` for unsupported types. `dst` must have at least
/// `elements` slots; `src` must hold at least `row_bytes(kind, elements)`
/// bytes.
pub fn dequantize_row(kind: u32, src: &[u8], dst: &mut [f32], elements: usize) -> bool {
    if !is_supported(kind) {
        return false;
    }
    match kind {
        F32 => {
            for i in 0..elements {
                dst[i] = f32le(src, i * 4);
            }
        }
        F16 => {
            for i in 0..elements {
                dst[i] = half_to_f32(u16le(src, i * 2));
            }
        }
        BF16 => {
            for i in 0..elements {
                dst[i] = bf16_to_f32(u16le(src, i * 2));
            }
        }
        Q4_0 => dequant_q4_0(src, dst, elements),
        Q4_1 => dequant_q4_1(src, dst, elements),
        Q5_0 => dequant_q5_0(src, dst, elements),
        Q5_1 => dequant_q5_1(src, dst, elements),
        Q8_0 => dequant_q8_0(src, dst, elements),
        Q8_1 => dequant_q8_1(src, dst, elements, detect_q8_1_layout(src, elements, 1)),
        Q2_K => dequant_q2_k(src, dst, elements),
        Q3_K => dequant_q3_k(src, dst, elements),
        Q4_K => dequant_q4_k(src, dst, elements),
        Q5_K => dequant_q5_k(src, dst, elements),
        Q6_K => dequant_q6_k(src, dst, elements),
        Q1_0 => dequant_q1_0(src, dst, elements),
        Q2_0 => dequant_q2_0(src, dst, elements),
        TQ1_0 => dequant_tq1_0(src, dst, elements),
        TQ2_0 => dequant_tq2_0(src, dst, elements),
        PQ2_0 => dequant_pq2_0(src, dst, elements),
        PQ2_0M => dequant_pq2_0m(src, dst, elements),
        PTQ1_0 => dequant_ptq1_0(src, dst, elements),
        _ => return false,
    }
    true
}

/// Prism PQ2_0: same 2-bit codec as Q2_0 but with a 128-wide group.
fn dequant_pq2_0(b: &[u8], y: &mut [f32], k: usize) {
    for blk in 0..k / 128 {
        let base = blk * 34;
        let d = half_to_f32(u16le(b, base));
        for j in 0..128 {
            let q = (b[base + 2 + j / 4] >> ((j % 4) * 2)) & 0x03;
            y[blk * 128 + j] = (q as i32 - 1) as f32 * d;
        }
    }
}

/// Prism PQ2_0M: `m + q * d`, used for KV-cache only.
fn dequant_pq2_0m(b: &[u8], y: &mut [f32], k: usize) {
    for blk in 0..k / 128 {
        let base = blk * 36;
        let d = half_to_f32(u16le(b, base));
        let m = half_to_f32(u16le(b, base + 2));
        for j in 0..128 {
            let q = (b[base + 4 + j / 4] >> ((j % 4) * 2)) & 0x03;
            y[blk * 128 + j] = m + q as f32 * d;
        }
    }
}

/// Prism PTQ1_0: base-3 trit packing at group size 128 (24 qs + 2 qh + f16 d).
fn dequant_ptq1_0(b: &[u8], y: &mut [f32], k: usize) {
    const POW3: [u32; 6] = [1, 3, 9, 27, 81, 243];
    for blk in 0..k / 128 {
        let base = blk * 28;
        let d = half_to_f32(u16le(b, base + 26));
        let mut out = blk * 128;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= 24 {
                for n in 0..5 {
                    for m in 0..c {
                        let q = b[base + j + m] as u32 * POW3[n];
                        let xi = ((q * 3) >> 8) as i32;
                        y[out] = (xi - 1) as f32 * d;
                        out += 1;
                    }
                }
                j += c;
            }
        }
        for n in 0..4 {
            for h in 0..2 {
                let q = b[base + 24 + h] as u32 * POW3[n];
                let xi = ((q * 3) >> 8) as i32;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
}

/// Q1_0: one sign bit per weight with a shared f16 scale.
fn dequant_q1_0(b: &[u8], y: &mut [f32], k: usize) {
    for blk in 0..k / 128 {
        let base = blk * 18;
        let d = half_to_f32(u16le(b, base));
        for j in 0..128 {
            let bit = (b[base + 2 + j / 8] >> (j % 8)) & 1;
            y[blk * 128 + j] = if bit != 0 { d } else { -d };
        }
    }
}

/// Q2_0: 2 bits per weight (`00=-1, 01=0, 10=+1, 11=+2`).
fn dequant_q2_0(b: &[u8], y: &mut [f32], k: usize) {
    for blk in 0..k / 64 {
        let base = blk * 18;
        let d = half_to_f32(u16le(b, base));
        for j in 0..64 {
            let q = (b[base + 2 + j / 4] >> ((j % 4) * 2)) & 0x03;
            y[blk * 64 + j] = (q as i32 - 1) as f32 * d;
        }
    }
}

/// TQ1_0: ternary weights packed five per byte (3^5 codebook) plus a 2-bit
/// tail. Layout: qs[48], qh[4], f16 d.
fn dequant_tq1_0(b: &[u8], y: &mut [f32], k: usize) {
    const POW3: [u32; 6] = [1, 3, 9, 27, 81, 243];
    for blk in 0..k / 256 {
        let base = blk * 54;
        let d = half_to_f32(u16le(b, base + 52));
        let mut out = blk * 256;
        for j in (0..48).step_by(32) {
            for n in 0..5 {
                for m in 0..32 {
                    let q = b[base + j + m] as u32 * POW3[n];
                    let xi = ((q * 3) >> 8) as i32;
                    y[out] = (xi - 1) as f32 * d;
                    out += 1;
                }
            }
        }
        for j in 32..48 {
            for n in 0..5 {
                let q = b[base + j] as u32 * POW3[n];
                let xi = ((q * 3) >> 8) as i32;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                let q = b[base + 48 + j] as u32 * POW3[n];
                let xi = ((q * 3) >> 8) as i32;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
}

/// TQ2_0: 2-bit ternary codes, 64 weights per f16 scale.
fn dequant_tq2_0(b: &[u8], y: &mut [f32], k: usize) {
    for blk in 0..k / 256 {
        let base = blk * 66;
        let d = half_to_f32(u16le(b, base + 64));
        for j in 0..64 {
            let q = (b[base + j / 4] >> ((j % 4) * 2)) & 0x03;
            let v = match q { 0 => 1.0f32, 1 => 2.0, 2 => 0.0, _ => -1.0 };
            y[blk * 256 + j] = v * d;
        }
    }
}

fn dequant_q4_0(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 32 {
        let o = i * 18;
        let d = half_to_f32(u16le(b, o));
        for j in 0..16 {
            let q = b[o + 2 + j];
            y[i * 32 + j] = ((q & 0x0f) as i32 - 8) as f32 * d;
            y[i * 32 + j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
        }
    }
}

fn dequant_q4_1(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 32 {
        let o = i * 20;
        let d = half_to_f32(u16le(b, o));
        let m = half_to_f32(u16le(b, o + 2));
        for j in 0..16 {
            let q = b[o + 4 + j];
            y[i * 32 + j] = (q & 0x0f) as f32 * d + m;
            y[i * 32 + j + 16] = (q >> 4) as f32 * d + m;
        }
    }
}

fn dequant_q5_0(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 32 {
        let o = i * 22;
        let d = half_to_f32(u16le(b, o));
        let qh = u32le(b, o + 2);
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q = b[o + 6 + j];
            y[i * 32 + j] = (((q & 0x0f) as u32 | xh0) as i32 - 16) as f32 * d;
            y[i * 32 + j + 16] = (((q >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
        }
    }
}

fn dequant_q5_1(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 32 {
        let o = i * 24;
        let d = half_to_f32(u16le(b, o));
        let m = half_to_f32(u16le(b, o + 2));
        let qh = u32le(b, o + 4);
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q = b[o + 8 + j];
            y[i * 32 + j] = ((q & 0x0f) as u32 | xh0) as f32 * d + m;
            y[i * 32 + j + 16] = ((q >> 4) as u32 | xh1) as f32 * d + m;
        }
    }
}

fn dequant_q8_0(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 32 {
        let o = i * 34;
        let d = half_to_f32(u16le(b, o));
        for j in 0..32 {
            y[i * 32 + j] = (b[o + 2 + j] as i8) as f32 * d;
        }
    }
}

/// Expand each bit of a byte into one byte per lane: bit j -> byte j == 1.
/// Used to build Q5_0 high-bit vectors without a scalar loop.
const fn bit_spread_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut bits = 0u64;
        let mut j = 0usize;
        while j < 8 {
            if i & (1 << j) != 0 {
                bits |= 1u64 << (j * 8);
            }
            j += 1;
        }
        table[i] = bits;
        i += 1;
    }
    table
}

static BIT_SPREAD: [u64; 256] = bit_spread_table();

/// Build two 16-byte vectors of 0/1 high bits for a Q5_0 super-block.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn q5_high_bits(qh: u32) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    let h0 = BIT_SPREAD[(qh & 0xff) as usize] as u128
        | ((BIT_SPREAD[((qh >> 8) & 0xff) as usize] as u128) << 64);
    let h1 = BIT_SPREAD[((qh >> 16) & 0xff) as usize] as u128
        | ((BIT_SPREAD[((qh >> 24) & 0xff) as usize] as u128) << 64);
    (
        _mm_loadu_si128(&h0 as *const u128 as *const __m128i),
        _mm_loadu_si128(&h1 as *const u128 as *const __m128i),
    )
}

/// Fused dequantize-and-dot kernels for the most common 4/8-bit types.
/// Returns `None` when the CPU or type is unsupported.
pub fn dot_quant(kind: u32, row: &[u8], x: &[f32], k: usize, _layout: Q8Layout) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            let avx = std::is_x86_feature_detected!("avx");
            // SAFETY: the target features are checked above. All kernels only
            // read `row`/`x` within `k` elements.
            unsafe {
                return match kind {
                    Q8_0 if avx => Some(dot_q8_0_avx(row, x, k)),
                    Q5_0 if avx => Some(dot_q5_0_avx(row, x, k)),
                    Q8_0 => Some(dot_q8_0_sse(row, x, k)),
                    Q5_0 => Some(dot_q5_0_sse(row, x, k)),
                    Q4_K => Some(dot_q4_k_sse(row, x, k)),
                    Q6_K if avx => Some(dot_q6_k_avx(row, x, k)),
                    Q6_K => Some(dot_q6_k_sse(row, x, k)),
                    _ => None,
                };
            }
        }
    }
    None
}

/// Q6_K: 16 sub-blocks of 16 values, 6-bit quants with 8-bit sub-block scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_q6_k_sse(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_ps();
    let mask0f = _mm_set1_epi8(0x0f);
    let mask03 = _mm_set1_epi8(0x03);
    let thirty_two = _mm_set1_epi32(32);

    for i in 0..k / 256 {
        let o = i * 210;
        let d = half_to_f32(u16le(row, o + 208));
        let ql = row.as_ptr().add(o);
        let qh = row.as_ptr().add(o + 128);
        let sc = row.as_ptr().add(o + 192);
        for n in 0..2 {
            let ql = ql.add(n * 64);
            let qh = qh.add(n * 32);
            let sc = sc.add(n * 8);
            for group in 0..8 {
                let l = group * 4;
                let is = l / 16;
                let a = std::ptr::read_unaligned(ql.add(l) as *const i32);
                let b = std::ptr::read_unaligned(ql.add(l + 32) as *const i32);
                let h = std::ptr::read_unaligned(qh.add(l) as *const i32);
                let va = _mm_cvtsi32_si128(a);
                let vb = _mm_cvtsi32_si128(b);
                let vh = _mm_cvtsi32_si128(h);

                let h1 = _mm_and_si128(vh, mask03);
                let h2 = _mm_and_si128(_mm_srli_epi16(vh, 2), mask03);
                let h3 = _mm_and_si128(_mm_srli_epi16(vh, 4), mask03);
                let h4 = _mm_and_si128(_mm_srli_epi16(vh, 6), mask03);

                let lo1 = _mm_and_si128(va, mask0f);
                let hi1 = _mm_and_si128(_mm_srli_epi16(va, 4), mask0f);
                let lo2 = _mm_and_si128(vb, mask0f);
                let hi2 = _mm_and_si128(_mm_srli_epi16(vb, 4), mask0f);

                let q1 = _mm_sub_epi32(
                    _mm_cvtepu8_epi32(_mm_or_si128(lo1, _mm_slli_epi16(h1, 4))),
                    thirty_two,
                );
                let q2 = _mm_sub_epi32(
                    _mm_cvtepu8_epi32(_mm_or_si128(lo2, _mm_slli_epi16(h2, 4))),
                    thirty_two,
                );
                let q3 = _mm_sub_epi32(
                    _mm_cvtepu8_epi32(_mm_or_si128(hi1, _mm_slli_epi16(h3, 4))),
                    thirty_two,
                );
                let q4 = _mm_sub_epi32(
                    _mm_cvtepu8_epi32(_mm_or_si128(hi2, _mm_slli_epi16(h4, 4))),
                    thirty_two,
                );

                let s1 = d * (*sc.add(is) as i8) as f32;
                let s2 = d * (*sc.add(is + 2) as i8) as f32;
                let s3 = d * (*sc.add(is + 4) as i8) as f32;
                let s4 = d * (*sc.add(is + 6) as i8) as f32;
                let base = i * 256 + n * 128 + l;
                let x1 = _mm_loadu_ps(x.as_ptr().add(base));
                let x2 = _mm_loadu_ps(x.as_ptr().add(base + 32));
                let x3 = _mm_loadu_ps(x.as_ptr().add(base + 64));
                let x4 = _mm_loadu_ps(x.as_ptr().add(base + 96));
                acc = _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(q1), _mm_set1_ps(s1)), x1),
                    acc,
                );
                acc = _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(q2), _mm_set1_ps(s2)), x2),
                    acc,
                );
                acc = _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(q3), _mm_set1_ps(s3)), x3),
                    acc,
                );
                acc = _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(q4), _mm_set1_ps(s4)), x4),
                    acc,
                );
            }
        }
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

/// AVX variant of the Q6_K kernel: two 4-lane groups joined per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse4.1")]
unsafe fn dot_q6_k_avx(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mask0f = _mm_set1_epi8(0x0f);
    let mask03 = _mm_set1_epi8(0x03);
    let thirty_two = _mm_set1_epi32(32);

    for i in 0..k / 256 {
        let o = i * 210;
        let d = half_to_f32(u16le(row, o + 208));
        let ql = row.as_ptr().add(o);
        let qh = row.as_ptr().add(o + 128);
        let sc = row.as_ptr().add(o + 192);
        for n in 0..2 {
            let ql = ql.add(n * 64);
            let qh = qh.add(n * 32);
            let sc = sc.add(n * 8);
            for g in 0..4 {
                let is = (g * 8) / 16;
                let mut qv = [[_mm_setzero_ps(); 4]; 2];
                for half in 0..2 {
                    let l = g * 8 + half * 4;
                    let a = std::ptr::read_unaligned(ql.add(l) as *const i32);
                    let b = std::ptr::read_unaligned(ql.add(l + 32) as *const i32);
                    let h = std::ptr::read_unaligned(qh.add(l) as *const i32);
                    let va = _mm_cvtsi32_si128(a);
                    let vb = _mm_cvtsi32_si128(b);
                    let vh = _mm_cvtsi32_si128(h);
                    let h1 = _mm_and_si128(vh, mask03);
                    let h2 = _mm_and_si128(_mm_srli_epi16(vh, 2), mask03);
                    let h3 = _mm_and_si128(_mm_srli_epi16(vh, 4), mask03);
                    let h4 = _mm_and_si128(_mm_srli_epi16(vh, 6), mask03);
                    let lo1 = _mm_and_si128(va, mask0f);
                    let hi1 = _mm_and_si128(_mm_srli_epi16(va, 4), mask0f);
                    let lo2 = _mm_and_si128(vb, mask0f);
                    let hi2 = _mm_and_si128(_mm_srli_epi16(vb, 4), mask0f);
                    let q1 = _mm_sub_epi32(
                        _mm_cvtepu8_epi32(_mm_or_si128(lo1, _mm_slli_epi16(h1, 4))),
                        thirty_two,
                    );
                    let q2 = _mm_sub_epi32(
                        _mm_cvtepu8_epi32(_mm_or_si128(lo2, _mm_slli_epi16(h2, 4))),
                        thirty_two,
                    );
                    let q3 = _mm_sub_epi32(
                        _mm_cvtepu8_epi32(_mm_or_si128(hi1, _mm_slli_epi16(h3, 4))),
                        thirty_two,
                    );
                    let q4 = _mm_sub_epi32(
                        _mm_cvtepu8_epi32(_mm_or_si128(hi2, _mm_slli_epi16(h4, 4))),
                        thirty_two,
                    );
                    qv[half] = [
                        _mm_cvtepi32_ps(q1),
                        _mm_cvtepi32_ps(q2),
                        _mm_cvtepi32_ps(q3),
                        _mm_cvtepi32_ps(q4),
                    ];
                }
                let base = i * 256 + n * 128 + g * 8;
                for k in 0..4 {
                    let s = d * (*sc.add(is + k * 2) as i8) as f32;
                    let q = _mm256_mul_ps(join_ps(qv[0][k], qv[1][k]), _mm256_set1_ps(s));
                    acc = _mm256_add_ps(
                        _mm256_mul_ps(q, _mm256_loadu_ps(x.as_ptr().add(base + k * 32))),
                        acc,
                    );
                }
            }
        }
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

/// Q4_K: 8 sub-blocks of 32 values with 6-bit packed scales/minimums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_q4_k_sse(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_ps();
    let mask0f = _mm_set1_epi8(0x0f);

    for i in 0..k / 256 {
        let o = i * 144;
        let d = half_to_f32(u16le(row, o));
        let dmin = half_to_f32(u16le(row, o + 2));
        let scales = &row[o + 4..o + 16];
        let mut qs = row.as_ptr().add(o + 16);
        let mut is = 0usize;
        for j in 0..4 {
            let (s1, m1) = get_scale_min_k4(is, scales);
            let (s2, m2) = get_scale_min_k4(is + 1, scales);
            let f1 = _mm_set1_ps(d * s1 as f32);
            let f2 = _mm_set1_ps(d * s2 as f32);
            let v1 = _mm_set1_ps(dmin * m1 as f32);
            let v2 = _mm_set1_ps(dmin * m2 as f32);
            for g in 0..8 {
                let l = g * 4;
                let q = std::ptr::read_unaligned(qs.add(l) as *const i32);
                let vq = _mm_cvtsi32_si128(q);
                let lo = _mm_cvtepu8_epi32(_mm_and_si128(vq, mask0f));
                let hi = _mm_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(vq, 4), mask0f));
                let y1 = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(lo), f1), v1);
                let y2 = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(hi), f2), v2);
                let base = i * 256 + j * 64 + l;
                acc = _mm_add_ps(_mm_mul_ps(y1, _mm_loadu_ps(x.as_ptr().add(base))), acc);
                acc = _mm_add_ps(_mm_mul_ps(y2, _mm_loadu_ps(x.as_ptr().add(base + 32))), acc);
            }
            qs = qs.add(32);
            is += 2;
        }
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

/// Combine two 128-bit float vectors into one 256-bit vector.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn join_ps(
    a: std::arch::x86_64::__m128,
    b: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse4.1")]
unsafe fn dot_q8_0_avx(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    for i in 0..k / 32 {
        let o = i * 34;
        let dv = _mm256_set1_ps(half_to_f32(u16le(row, o)));
        let qs = row.as_ptr().add(o + 2);
        for j in 0..4 {
            let a = std::ptr::read_unaligned(qs.add(j * 8) as *const i32);
            let b = std::ptr::read_unaligned(qs.add(j * 8 + 4) as *const i32);
            let v0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(a)));
            let v1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(b)));
            let qf = _mm256_mul_ps(join_ps(v0, v1), dv);
            let xf = _mm256_loadu_ps(x.as_ptr().add(i * 32 + j * 8));
            acc = _mm256_add_ps(_mm256_mul_ps(qf, xf), acc);
        }
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse4.1")]
unsafe fn dot_q5_0_avx(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mask0f = _mm_set1_epi8(0x0f);
    let sixteen = _mm_set1_epi32(16);

    for i in 0..k / 32 {
        let o = i * 22;
        let dv = _mm256_set1_ps(half_to_f32(u16le(row, o)));
        let (h0all, h1all) = q5_high_bits(u32le(row, o + 2));
        let qs = row.as_ptr().add(o + 6);
        for half in 0..2 {
            let h0h = if half == 0 { h0all } else { _mm_srli_si128(h0all, 8) };
            let h1h = if half == 0 { h1all } else { _mm_srli_si128(h1all, 8) };
            let mut q0v = [_mm_setzero_ps(); 2];
            let mut q1v = [_mm_setzero_ps(); 2];
            for group in 0..2 {
                let base = half * 8 + group * 4;
                let qb = std::ptr::read_unaligned(qs.add(base) as *const i32);
                let qv = _mm_cvtsi32_si128(qb);
                let lo = _mm_cvtepu8_epi32(_mm_and_si128(qv, mask0f));
                let hi = _mm_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qv, 4), mask0f));
                let hv0 = if group == 0 { h0h } else { _mm_srli_si128(h0h, 4) };
                let hv1 = if group == 0 { h1h } else { _mm_srli_si128(h1h, 4) };
                let a = _mm_sub_epi32(
                    _mm_or_si128(lo, _mm_slli_epi32(_mm_cvtepu8_epi32(hv0), 4)),
                    sixteen,
                );
                let b = _mm_sub_epi32(
                    _mm_or_si128(hi, _mm_slli_epi32(_mm_cvtepu8_epi32(hv1), 4)),
                    sixteen,
                );
                q0v[group] = _mm_cvtepi32_ps(a);
                q1v[group] = _mm_cvtepi32_ps(b);
            }
            let q0 = join_ps(q0v[0], q0v[1]);
            let q1 = join_ps(q1v[0], q1v[1]);
            let x0 = _mm256_loadu_ps(x.as_ptr().add(i * 32 + half * 8));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(i * 32 + 16 + half * 8));
            acc = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(q0, dv), x0), acc);
            acc = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(q1, dv), x1), acc);
        }
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_q8_0_sse(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_ps();
    for i in 0..k / 32 {
        let o = i * 34;
        let d = half_to_f32(u16le(row, o));
        let dv = _mm_set1_ps(d);
        let qs = row.as_ptr().add(o + 2);
        for j in 0..8 {
            let q4 = std::ptr::read_unaligned(qs.add(j * 4) as *const i32);
            let q32 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(q4));
            let qf = _mm_mul_ps(_mm_cvtepi32_ps(q32), dv);
            let xf = _mm_loadu_ps(x.as_ptr().add(i * 32 + j * 4));
            acc = _mm_add_ps(_mm_mul_ps(qf, xf), acc);
        }
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_q5_0_sse(row: &[u8], x: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_ps();
    let mask0f = _mm_set1_epi8(0x0f);
    let sixteen = _mm_set1_epi32(16);

    for i in 0..k / 32 {
        let o = i * 22;
        let dv = _mm_set1_ps(half_to_f32(u16le(row, o)));
        let (h0all, h1all) = q5_high_bits(u32le(row, o + 2));
        let qs = row.as_ptr().add(o + 6);
        for half in 0..2 {
            let h0h = if half == 0 { h0all } else { _mm_srli_si128(h0all, 8) };
            let h1h = if half == 0 { h1all } else { _mm_srli_si128(h1all, 8) };
            for group in 0..2 {
                let base = half * 8 + group * 4;
                let qv = _mm_cvtsi32_si128(std::ptr::read_unaligned(qs.add(base) as *const i32));
                let lo = _mm_cvtepu8_epi32(_mm_and_si128(qv, mask0f));
                let hi = _mm_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qv, 4), mask0f));
                let hv0 = if group == 0 { h0h } else { _mm_srli_si128(h0h, 4) };
                let hv1 = if group == 0 { h1h } else { _mm_srli_si128(h1h, 4) };
                let q0 = _mm_sub_epi32(
                    _mm_or_si128(lo, _mm_slli_epi32(_mm_cvtepu8_epi32(hv0), 4)),
                    sixteen,
                );
                let q1 = _mm_sub_epi32(
                    _mm_or_si128(hi, _mm_slli_epi32(_mm_cvtepu8_epi32(hv1), 4)),
                    sixteen,
                );
                let x0 = _mm_loadu_ps(x.as_ptr().add(i * 32 + base));
                let x1 = _mm_loadu_ps(x.as_ptr().add(i * 32 + 16 + base));
                let f0 = _mm_mul_ps(_mm_cvtepi32_ps(q0), dv);
                let f1 = _mm_mul_ps(_mm_cvtepi32_ps(q1), dv);
                acc = _mm_add_ps(_mm_mul_ps(f0, x0), acc);
                acc = _mm_add_ps(_mm_mul_ps(f1, x1), acc);
            }
        }
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize a row with a pre-detected Q8_1 field layout.
pub fn dequantize_row_layout(
    kind: u32,
    src: &[u8],
    dst: &mut [f32],
    elements: usize,
    layout: Q8Layout,
) -> bool {
    if kind == Q8_1 {
        return dequantize_q8_1(src, dst, elements, layout);
    }
    dequantize_row(kind, src, dst, elements)
}

/// Dequantize one Q8_1 row with an explicitly chosen field layout.
pub fn dequantize_q8_1(b: &[u8], y: &mut [f32], k: usize, layout: Q8Layout) -> bool {
    for i in 0..k / 32 {
        let o = i * 36;
        let d = half_to_f32(u16le(b, o));
        let qs = match layout {
            Q8Layout::Old => &b[o + 2..o + 34],
            Q8Layout::New => &b[o + 4..o + 36],
        };
        for j in 0..32 {
            y[i * 32 + j] = (qs[j] as i8) as f32 * d;
        }
    }
    true
}

fn dequant_q8_1(b: &[u8], y: &mut [f32], k: usize, layout: Q8Layout) {
    let _ = dequantize_q8_1(b, y, k, layout);
}

/// Detect the Q8_1 field layout by sampling rows throughout a tensor.
///
/// The correct interpretation satisfies `s = d * sum(qs)`; the wrong one
/// usually produces NaN/Inf half words or a huge residual. We use the median
/// of the finite residuals to be robust against individual broken blocks.
pub fn detect_q8_1_layout(bytes: &[u8], cols: usize, rows: usize) -> Q8Layout {
    let row_len = cols.div_ceil(32) * 36;
    if row_len == 0 || bytes.len() < row_len {
        return Q8Layout::New;
    }
    let samples = rows.min(16);
    let mut old_errors = Vec::new();
    let mut new_errors = Vec::new();
    for s in 0..samples {
        let r = if samples == 1 { 0 } else { s * rows / samples };
        let start = r * row_len;
        if start + row_len > bytes.len() {
            break;
        }
        let b = &bytes[start..start + row_len];
        let blocks = (cols / 32).min(32);
        for i in 0..blocks {
            let o = i * 36;
            let d = half_to_f32(u16le(b, o)) as f64;
            if d == 0.0 || !d.is_finite() {
                continue;
            }
            let s_old = half_to_f32(u16le(b, o + 34)) as f64;
            let s_new = half_to_f32(u16le(b, o + 2)) as f64;
            let sum_old: i64 = b[o + 2..o + 34]
                .iter()
                .map(|&q| q as i8 as i64)
                .sum();
            let sum_new: i64 = b[o + 4..o + 36]
                .iter()
                .map(|&q| q as i8 as i64)
                .sum();
            let e_old = (s_old / d - sum_old as f64).abs();
            let e_new = (s_new / d - sum_new as f64).abs();
            // Wrong-layout candidates are frequently NaN/Inf half words;
            // those must not poison the comparison.
            if e_old.is_finite() {
                old_errors.push(e_old);
            }
            if e_new.is_finite() {
                new_errors.push(e_new);
            }
        }
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::INFINITY;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    if median(&mut old_errors) < median(&mut new_errors) {
        Q8Layout::Old
    } else {
        Q8Layout::New
    }
}

fn dequant_q2_k(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 256 {
        let o = i * 84;
        let d = half_to_f32(u16le(b, o + 80));
        let dmin = half_to_f32(u16le(b, o + 82));
        let scales = &b[o..o + 16];
        let qs = &b[o + 16..o + 80];
        let mut is = 0usize;
        let mut out = i * 256;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0f) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    y[out + l] = dl * ((qs[l] >> shift) & 3) as f32 - ml;
                }
                out += 16;
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0f) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    y[out + l] = dl * ((qs[l + 16] >> shift) & 3) as f32 - ml;
                }
                out += 16;
                shift += 2;
            }
        }
    }
}

fn dequant_q3_k(b: &[u8], y: &mut [f32], k: usize) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    for i in 0..k / 256 {
        let o = i * 110;
        let d = half_to_f32(u16le(b, o + 108));
        let hm = &b[o..o + 32];
        let qs = &b[o + 32..o + 96];
        let sc_bytes = &b[o + 96..o + 108];

        let mut aux = [
            u32le(sc_bytes, 0),
            u32le(sc_bytes, 4),
            u32le(sc_bytes, 8),
            0u32,
        ];
        let tmp = aux[2];
        aux[2] = (aux[0] >> 4) & KMASK2 | ((tmp >> 4) & KMASK1) << 4;
        aux[3] = (aux[1] >> 4) & KMASK2 | ((tmp >> 6) & KMASK1) << 4;
        aux[0] = (aux[0] & KMASK2) | ((tmp >> 0) & KMASK1) << 4;
        aux[1] = (aux[1] & KMASK2) | ((tmp >> 2) & KMASK1) << 4;
        let mut scales = [0i8; 16];
        for j in 0..16 {
            scales[j] = ((aux[j / 4] >> ((j % 4) * 8)) & 0xff) as u8 as i8;
        }

        let mut m = 1u8;
        let mut is = 0usize;
        let mut out = i * 256;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 64];
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let hi = if hm[l] & m != 0 { 0 } else { 4 };
                    y[out + l] = dl * (((q[l] >> shift) & 3) as i32 - hi) as f32;
                }
                out += 16;
                let dl = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let hi = if hm[l + 16] & m != 0 { 0 } else { 4 };
                    y[out + l] = dl * (((q[l + 16] >> shift) & 3) as i32 - hi) as f32;
                }
                out += 16;
                shift += 2;
                m <<= 1;
            }
        }
    }
}

fn dequant_q4_k(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 256 {
        let o = i * 144;
        let d = half_to_f32(u16le(b, o));
        let dmin = half_to_f32(u16le(b, o + 2));
        let scales = &b[o + 4..o + 16];
        let mut qs = &b[o + 16..o + 144];
        let mut is = 0usize;
        let mut out = i * 256;
        for _j in 0..4 {
            let (s1, m1) = get_scale_min_k4(is, scales);
            let (s2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * s1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * s2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                y[out + l] = d1 * (qs[l] & 0x0f) as f32 - m1;
            }
            out += 32;
            for l in 0..32 {
                y[out + l] = d2 * (qs[l] >> 4) as f32 - m2;
            }
            out += 32;
            qs = &qs[32..];
            is += 2;
        }
    }
}

fn dequant_q5_k(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 256 {
        let o = i * 176;
        let d = half_to_f32(u16le(b, o));
        let dmin = half_to_f32(u16le(b, o + 2));
        let scales = &b[o + 4..o + 16];
        let qh = &b[o + 16..o + 48];
        let mut ql = &b[o + 48..o + 176];
        let mut is = 0usize;
        let mut out = i * 256;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _j in 0..4 {
            let (s1, m1) = get_scale_min_k4(is, scales);
            let (s2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * s1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * s2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let q = (ql[l] & 0x0f) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                y[out + l] = d1 * q - m1;
            }
            out += 32;
            for l in 0..32 {
                let q = (ql[l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                y[out + l] = d2 * q - m2;
            }
            out += 32;
            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequant_q6_k(b: &[u8], y: &mut [f32], k: usize) {
    for i in 0..k / 256 {
        let o = i * 210;
        let d = half_to_f32(u16le(b, o + 208));
        let ql0 = &b[o..o + 128];
        let qh0 = &b[o + 128..o + 192];
        let scales = &b[o + 192..o + 208];
        let mut out = i * 256;
        for n in 0..2 {
            let ql = &ql0[n * 64..n * 64 + 64];
            let qh = &qh0[n * 32..n * 32 + 32];
            let sc = &scales[n * 8..n * 8 + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | ((qh[l] >> 0) & 3) << 4) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | ((qh[l] >> 2) & 3) << 4) as i8 - 32;
                let q3 = ((ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) as i8 - 32;
                y[out + l] = d * sc[is] as i8 as f32 * q1 as f32;
                y[out + 32 + l] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[out + 64 + l] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[out + 96 + l] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            out += 128;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(half_to_f32(0x0000), 0.0);
    }

    #[test]
    fn block_layouts() {
        assert_eq!(block_bytes(Q4_0), 18);
        assert_eq!(block_bytes(Q5_0), 22);
        assert_eq!(block_bytes(Q8_0), 34);
        assert_eq!(block_bytes(Q2_K), 4 + 16 + 64);
        assert_eq!(block_bytes(Q3_K), 2 + 32 + 64 + 12);
        assert_eq!(block_bytes(Q4_K), 4 + 12 + 128);
        assert_eq!(block_bytes(Q5_K), 4 + 12 + 32 + 128);
        assert_eq!(block_bytes(Q6_K), 2 + 16 + 64 + 128);
    }

    #[test]
    fn fused_dot_matches_dequant_then_dot() {
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..256).map(|_| next()).collect();
        for kind in [Q8_0, Q5_0] {
            let mut row = vec![0u8; row_bytes_for(kind, 256)];
            for b in &mut row {
                *b = (next() * 100.0) as i8 as u8;
            }
            // Make the scale finite and non-zero.
            row[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
            let mut deq = vec![0f32; 256];
            dequantize_row(kind, &row, &mut deq, 256);
            let expected: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = dot_quant(kind, &row, &x, 256, Q8Layout::New).unwrap();
            if expected.is_finite() {
                assert!(
                    (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "kind {kind}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[cfg(test)]
    fn row_bytes_for(kind: u32, n: usize) -> usize {
        n.div_ceil(block_elements(kind)) * block_bytes(kind)
    }

    #[test]
    fn fused_dot_k_quants_match_reference() {
        let mut rng = 777u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..256).map(|_| next()).collect();
        for (kind, block) in [(Q4_K, 144usize), (Q6_K, 210usize)] {
            let mut row = vec![0u8; 256 / 256 * block];
            for b in &mut row {
                *b = (next() * 100.0) as i8 as u8;
            }
            // Valid, finite scales for every block.
            for off in (0..row.len()).step_by(block) {
                let d_off = if kind == Q4_K { off } else { off + 208 };
                row[d_off..d_off + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                if kind == Q4_K {
                    row[off + 2..off + 4].copy_from_slice(&0x3000u16.to_le_bytes());
                }
            }
            let mut deq = vec![0f32; 256];
            assert!(dequantize_row(kind, &row, &mut deq, 256));
            let expected: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = dot_quant(kind, &row, &x, 256, Q8Layout::New).unwrap();
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "kind {kind}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn q8_1_layout_detection_old_gguf() {
        // First block of `output.weight` from stories15M-q4_0.gguf.
        let b = [
            217u8, 18, 184, 239, 127, 26, 251, 248, 11, 35, 29, 24, 233, 248, 22, 239, 227, 29,
            22, 15, 1, 0, 255, 52, 15, 250, 11, 51, 8, 152, 6, 19, 12, 225, 204, 26,
        ];
        assert!(matches!(detect_q8_1_layout(&b, 32, 1), Q8Layout::Old));
        let mut y = [0f32; 32];
        dequantize_row(Q8_1, &b, &mut y, 32);
        assert!((y[0] - -0.06018).abs() < 1e-4);
        assert!((y[2] - 0.10616).abs() < 1e-4);
    }

    #[test]
    fn q4_0_roundtrip_of_known_block() {
        // d = -0.5 (half 0xb800), nibble 8 -> 0, nibble 7 -> -1
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&0xb800u16.to_le_bytes());
        for j in 0..16 {
            b[2 + j] = 0x78; // low = 8 -> 0, high = 7 -> -1
        }
        let mut y = vec![0f32; 32];
        assert!(dequantize_row(Q4_0, &b, &mut y, 32));
        assert_eq!(y[0], 0.0);
        assert_eq!(y[16], 0.5);
    }
}
