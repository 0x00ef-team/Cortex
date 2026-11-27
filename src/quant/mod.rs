//! GGML quantization types and their dequantization routines.
//!
//! Every routine in this module is a direct, dependency-free port of the
//! reference implementations in `ggml-quants.c`.  We dequantize one row at a
//! time (into a scratch buffer) so that large models stay in their compact
//! quantized representation in RAM.

mod dequant;

pub use dequant::{
    block_bytes, block_elements, dequantize_row, dequantize_row_layout, detect_q8_1_layout,
    dot_quant, is_supported, type_name, Q8Layout,
};

/// Numeric identifiers of GGML tensor types, as stored in GGUF files.
pub mod ggml_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q8_1: u32 = 9;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const Q8_K: u32 = 15;
    pub const IQ2_XXS: u32 = 16;
    pub const IQ2_XS: u32 = 17;
    pub const IQ3_XXS: u32 = 18;
    pub const IQ1_S: u32 = 19;
    pub const IQ4_NL: u32 = 20;
    pub const IQ3_S: u32 = 21;
    pub const IQ2_S: u32 = 22;
    pub const IQ4_XS: u32 = 23;
    pub const I8: u32 = 24;
    pub const I16: u32 = 25;
    pub const I32: u32 = 26;
    pub const I64: u32 = 27;
    pub const F64: u32 = 28;
    pub const IQ1_M: u32 = 29;
    pub const BF16: u32 = 30;
    pub const TQ1_0: u32 = 34;
    pub const TQ2_0: u32 = 35;
    pub const Q1_0: u32 = 41;
    pub const Q2_0: u32 = 42;
    pub const PQ2_0: u32 = 142;
    pub const PTQ1_0: u32 = 143;
    pub const PQ2_0M: u32 = 145;
}

/// Number of bytes needed to store `elements` values of a given type.
pub fn row_bytes(kind: u32, elements: usize) -> usize {
    let n = block_elements(kind);
    let b = block_bytes(kind);
    let blocks = elements.div_ceil(n);
    blocks * b
}
