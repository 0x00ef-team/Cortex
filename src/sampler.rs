//! Token sampling: greedy, temperature, top-k, top-p, min-p and repetition
//! penalty. Includes a tiny xorshift RNG so the crate stays dependency-free.

use crate::error::{Error, Result};

/// Deterministic pseudo-random generator (xorshift64*).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    /// 0 disables top-k filtering.
    pub top_k: usize,
    /// 1.0 disables nucleus filtering.
    pub top_p: f32,
    pub min_p: f32,
    /// 1.0 disables repetition penalty.
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    rng: Rng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_window: 64,
            rng: Rng::new(seed),
        }
    }

    pub fn greedy(seed: u64) -> Self {
        let mut s = Self::new(seed);
        s.temperature = 0.0;
        s
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Pick the next token from a vector of logits.
    pub fn sample(&mut self, logits: &[f32], generated: &[u32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(Error::InvalidTokenizer("empty logits tensor".into()));
        }

        let mut l = logits.to_vec();

        if self.repeat_penalty > 1.0 && !generated.is_empty() {
            let start = generated.len().saturating_sub(self.repeat_window);
            for &id in &generated[start..] {
                if let Some(v) = l.get_mut(id as usize) {
                    if *v > 0.0 {
                        *v /= self.repeat_penalty;
                    } else {
                        *v *= self.repeat_penalty;
                    }
                }
            }
        }

        if self.temperature <= 0.0 {
            return Ok(argmax(&l));
        }

        let inv_t = 1.0 / self.temperature;
        let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Ok(argmax(&l));
        }
        let mut probs: Vec<f32> = l
            .iter()
            .map(|&v| ((v - max) * inv_t).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Ok(argmax(&l));
        }
        for p in &mut probs {
            *p /= sum;
        }

        let max_p = probs.iter().cloned().fold(0.0f32, f32::max);
        let min_threshold = self.min_p * max_p;

        // Candidates sorted by descending probability.
        let mut order: Vec<usize> = (0..probs.len())
            .filter(|&i| probs[i] >= min_threshold && probs[i] > 0.0)
            .collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        if self.top_k > 0 && order.len() > self.top_k {
            order.truncate(self.top_k);
        }
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let target = self.top_p;
            let total: f32 = order.iter().map(|&i| probs[i]).sum();
            if total > 0.0 {
                let mut cum = 0.0;
                let mut keep = order.len();
                for (j, &i) in order.iter().enumerate() {
                    cum += probs[i] / total;
                    if cum >= target {
                        keep = j + 1;
                        break;
                    }
                }
                order.truncate(keep);
            }
        }

        if order.is_empty() {
            return Ok(argmax(&l));
        }

        let total: f64 = order.iter().map(|&i| probs[i] as f64).sum();
        let mut r = self.rng.next_f64() * total;
        let mut chosen = *order.last().unwrap();
        for &i in &order {
            r -= probs[i] as f64;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        Ok(chosen as u32)
    }
}

#[inline]
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::greedy(42);
        assert_eq!(s.sample(&[0.1, 5.0, -2.0], &[]).unwrap(), 1);
    }

    #[test]
    fn top_k_restricts_choices() {
        let mut s = Sampler::new(7);
        s.temperature = 1.0;
        s.top_k = 1;
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.0, 10.0, 1.0], &[]).unwrap(), 1);
        }
    }

    #[test]
    fn repetition_penalty_changes_preference() {
        let mut s = Sampler::greedy(1);
        s.repeat_penalty = 10.0;
        s.repeat_window = 10;
        // Without penalty token 0 wins; with penalty token 1 should.
        assert_eq!(s.sample(&[2.0, 1.0], &[0]).unwrap(), 1);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
