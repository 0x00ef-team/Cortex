//! GGUF metadata values. Strings are kept as raw bytes because tokenizer
//! vocabularies occasionally contain byte sequences that are not valid UTF-8.

use std::borrow::Cow;

/// Numeric tags for GGUF value types.
pub mod gguf_value_type {
    pub const U8: u32 = 0;
    pub const I8: u32 = 1;
    pub const U16: u32 = 2;
    pub const I16: u32 = 3;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const U64: u32 = 10;
    pub const I64: u32 = 11;
    pub const F64: u32 = 12;
}

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    /// Raw bytes of a UTF-8 (or byte-fallback) string.
    String(Vec<u8>),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// Bytes of a string value.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    /// Lossy UTF-8 view of a string value.
    pub fn str(&self) -> Option<Cow<'_, str>> {
        self.bytes().map(|b| String::from_utf8_lossy(b))
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U8(v) => Some(*v as u32),
            Value::U16(v) => Some(*v as u32),
            Value::U32(v) => Some(*v),
            Value::I32(v) if *v >= 0 => Some(*v as u32),
            Value::U64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            Value::U32(v) => Some(*v as f32),
            Value::U8(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            Value::U8(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Convenience: an array of raw byte strings.
    pub fn as_byte_strings(&self) -> Option<Vec<&[u8]>> {
        self.as_array().map(|a| {
            a.iter()
                .map(|v| v.bytes().unwrap_or(&[]))
                .collect()
        })
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        self.as_array()
            .map(|a| a.iter().filter_map(Value::as_f32).collect())
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        self.as_array()
            .map(|a| a.iter().filter_map(Value::as_u32).collect())
    }
}
