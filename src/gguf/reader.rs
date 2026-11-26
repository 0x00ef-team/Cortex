//! Binary reader for the GGUF container format (version 3).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};
use crate::quant::row_bytes;

use super::value::{gguf_value_type as vt, Value};

/// Descriptor of one tensor in a GGUF file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML dimensions: `dims[0]` is the fastest changing (columns).
    pub shape: Vec<u64>,
    /// GGML type id (see `crate::quant::ggml_type`).
    pub kind: u32,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn rows(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1] as usize
        } else {
            1
        }
    }

    pub fn cols(&self) -> usize {
        self.shape.first().copied().unwrap_or(0) as usize
    }

    pub fn byte_len(&self) -> Result<usize> {
        row_bytes(self.kind, self.cols())
            .checked_mul(self.rows())
            .ok_or_else(|| Error::InvalidGguf(format!("tensor {} is too large", self.name)))
    }
}

/// An opened GGUF file. Metadata is parsed eagerly; tensor data is not.
pub struct GgufFile {
    file: File,
    data_start: u64,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path.as_ref())?);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(Error::InvalidGguf("bad magic, expected GGUF".into()));
        }

        let version = read_u32(&mut reader)?;
        if version != 3 {
            return Err(Error::InvalidGguf(format!(
                "unsupported GGUF version {version} (expected 3)"
            )));
        }
        let tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;

        if tensor_count > 100_000 || kv_count > 100_000 {
            return Err(Error::InvalidGguf("unreasonably large header counts".into()));
        }

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key_bytes = read_string_bytes(&mut reader)?;
            let key = String::from_utf8_lossy(&key_bytes).into_owned();
            let kind = read_u32(&mut reader)?;
            let value = read_value(&mut reader, kind, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name_bytes = read_string_bytes(&mut reader)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();
            let n_dim = read_u32(&mut reader)?;
            if n_dim == 0 || n_dim > 8 {
                return Err(Error::InvalidGguf(format!(
                    "tensor {name} has {n_dim} dimensions"
                )));
            }
            let mut shape = Vec::with_capacity(n_dim as usize);
            for _ in 0..n_dim {
                shape.push(read_u64(&mut reader)?);
            }
            let kind = read_u32(&mut reader)?;
            let offset = read_u64(&mut reader)?;
            tensors.push(TensorInfo {
                name,
                shape,
                kind,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u32)
            .unwrap_or(32)
            .max(1) as u64;
        let pos = reader.stream_position()?;
        let data_start = pos.div_ceil(alignment) * alignment;

        let mut by_name = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            by_name.insert(t.name.clone(), i);
        }

        Ok(Self {
            file: File::open(path)?,
            data_start,
            version,
            metadata,
            tensors,
            by_name,
        })
    }

    /// Look up a tensor descriptor by name.
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.by_name
            .get(name)
            .map(|i| &self.tensors[*i])
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }

    /// Read the raw bytes of a tensor from the data section.
    pub fn tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self.info(name)?;
        let len = info.byte_len()?;
        self.file.seek(SeekFrom::Start(
            self.data_start
                .checked_add(info.offset)
                .ok_or_else(|| Error::InvalidGguf("tensor offset overflow".into()))?,
        ))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Convenience accessors for metadata.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(Value::as_u32)
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(Value::as_f32)
    }
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.metadata
            .get(key)
            .and_then(Value::str)
            .map(|c| c.into_owned())
    }
}

#[inline]
fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

#[inline]
fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[inline]
fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_string_bytes<R: Read>(r: &mut R) -> Result<Vec<u8>> {
    let len = read_u64(r)?;
    if len > 256 * 1024 * 1024 {
        return Err(Error::InvalidGguf("string is too long".into()));
    }
    let mut b = vec![0u8; len as usize];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_value<R: Read>(r: &mut R, kind: u32, depth: u32) -> Result<Value> {
    if depth > 1 {
        return Err(Error::InvalidGguf("nested GGUF arrays are not allowed".into()));
    }
    let v = match kind {
        vt::U8 => Value::U8(read_u8(r)?),
        vt::I8 => Value::I8(read_u8(r)? as i8),
        vt::U16 => Value::U16(read_u16(r)?),
        vt::I16 => Value::I16(read_u16(r)? as i16),
        vt::U32 => Value::U32(read_u32(r)?),
        vt::I32 => Value::I32(read_u32(r)? as i32),
        vt::F32 => Value::F32(read_f32(r)?),
        vt::BOOL => Value::Bool(read_u8(r)? != 0),
        vt::STRING => Value::String(read_string_bytes(r)?),
        vt::U64 => Value::U64(read_u64(r)?),
        vt::I64 => Value::I64(read_u64(r)? as i64),
        vt::F64 => Value::F64(read_f64(r)?),
        vt::ARRAY => {
            let base = read_u32(r)?;
            let count = read_u64(r)?;
            if count > 10_000_000 {
                return Err(Error::InvalidGguf("array is too long".into()));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(r, base, depth + 1)?);
            }
            Value::Array(items)
        }
        other => {
            return Err(Error::InvalidGguf(format!(
                "unknown metadata value type {other}"
            )))
        }
    };
    Ok(v)
}
