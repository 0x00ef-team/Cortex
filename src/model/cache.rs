//! Per-layer key/value cache.

/// Cached keys and values for one transformer layer.
///
/// Layout is `[capacity, n_kv_heads * head_dim]`, row-major so that a single
/// cached timestep is one contiguous slice.
pub struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    capacity: usize,
    width: usize,
}

impl LayerCache {
    pub fn new(capacity: usize, width: usize) -> Self {
        Self {
            keys: vec![0.0; capacity * width],
            values: vec![0.0; capacity * width],
            capacity,
            width,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Store the K/V vectors produced at position `pos`.
    ///
    /// The cache grows geometrically on demand, so we never allocate the
    /// full `n_ctx x layers` buffer up front.
    pub fn append(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        if pos >= self.capacity {
            let new_capacity = (pos + 1).max(self.capacity * 2).max(64);
            self.keys.resize(new_capacity * self.width, 0.0);
            self.values.resize(new_capacity * self.width, 0.0);
            self.capacity = new_capacity;
        }
        let start = pos * self.width;
        self.keys[start..start + self.width].copy_from_slice(k);
        self.values[start..start + self.width].copy_from_slice(v);
    }

    /// Keys for one timestep and one KV head.
    #[inline]
    pub fn key(&self, pos: usize, head: usize, head_dim: usize) -> &[f32] {
        let start = pos * self.width + head * head_dim;
        &self.keys[start..start + head_dim]
    }

    /// Values for one timestep and one KV head.
    #[inline]
    pub fn value(&self, pos: usize, head: usize, head_dim: usize) -> &[f32] {
        let start = pos * self.width + head * head_dim;
        &self.values[start..start + head_dim]
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }
}
