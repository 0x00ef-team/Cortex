//! A small BPE tokenizer that works directly from GGUF's `tokenizer.ggml.*`
//! metadata. No SentencePiece / tokenizers dependency is used.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::gguf::{GgufFile, Value};

/// Which pre-tokenization convention the vocabulary uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// SentencePiece-style: spaces are represented by U+2581 (`▁`).
    SentencePiece,
    /// GPT-2 / Qwen style: bytes are mapped to printable Unicode codepoints.
    ByteBpe,
}

const SPACE_MARK: char = '\u{2581}';
const SPACE_MARK_BYTES: &[u8] = b"\xe2\x96\x81";

/// Tokenizer built from a GGUF vocabulary.
pub struct Tokenizer {
    /// Original byte payload of every token.
    tokens: Vec<Vec<u8>>,
    scores: Vec<f32>,
    /// Map from token bytes (and alternate space spellings) to token id.
    lookup: HashMap<Vec<u8>, u32>,
    /// GPT-2 style explicit merge list: concatenation -> priority (lower first).
    merge_ranks: HashMap<Vec<u8>, usize>,
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
    pub add_bos: bool,
    pub add_eos: bool,
    pub pre: PreTokenizer,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Build a tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let m = &gguf.metadata;

        let token_values = m
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::InvalidTokenizer("missing tokenizer.ggml.tokens".into()))?;
        let tokens: Vec<Vec<u8>> = token_values
            .iter()
            .map(|v| v.bytes().unwrap_or(&[]).to_vec())
            .collect();
        if tokens.is_empty() {
            return Err(Error::InvalidTokenizer("empty vocabulary".into()));
        }

        let scores = m
            .get("tokenizer.ggml.scores")
            .and_then(Value::as_f32_array)
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        let mut scores = scores;
        scores.resize(tokens.len(), 0.0);

        let pre_name = gguf.get_str("tokenizer.ggml.pre").unwrap_or_default();
        let pre = if pre_name.contains("gpt2") || pre_name.contains("qwen") {
            PreTokenizer::ByteBpe
        } else {
            PreTokenizer::SentencePiece
        };

        // Build a lookup that understands both spellings of a leading space.
        // This makes us independent of whether the converter stored `▁` or
        // a plain ASCII space in the vocabulary.
        let mut lookup = HashMap::with_capacity(tokens.len() * 2);
        for (id, tok) in tokens.iter().enumerate() {
            let id = id as u32;
            lookup.insert(tok.clone(), id);
            if tok.starts_with(b" ") {
                let mut alt = SPACE_MARK_BYTES.to_vec();
                alt.extend_from_slice(&tok[1..]);
                lookup.entry(alt).or_insert(id);
            } else if tok.starts_with(SPACE_MARK_BYTES) {
                let mut alt = vec![b' '];
                alt.extend_from_slice(&tok[SPACE_MARK_BYTES.len()..]);
                lookup.entry(alt).or_insert(id);
            }
        }

        let bos = gguf.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        let unk = gguf
            .get_u32("tokenizer.ggml.unknown_token_id")
            .or_else(|| gguf.get_u32("tokenizer.ggml.unknown_token"))
            .unwrap_or(0);

        let add_bos = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let add_eos = gguf
            .metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        // Qwen/GPT-2 vocabularies carry an explicit, ordered merge list
        // instead of scores. Earlier merges have higher priority.
        let mut merge_ranks = HashMap::new();
        if let Some(merges) = m
            .get("tokenizer.ggml.merges")
            .and_then(Value::as_array)
        {
            for (rank, item) in merges.iter().enumerate() {
                let Some(s) = item.bytes() else { continue };
                let Some(sep) = s.iter().position(|&b| b == b' ') else { continue };
                let mut pair = s[..sep].to_vec();
                pair.extend_from_slice(&s[sep + 1..]);
                merge_ranks.entry(pair).or_insert(rank);
            }
        }

        let (byte_to_unicode, unicode_to_byte) = byte_unicode_tables();

        Ok(Self {
            tokens,
            scores,
            lookup,
            merge_ranks,
            bos,
            eos,
            unk,
            add_bos,
            add_eos,
            pre,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Raw bytes of a token id (before byte<->unicode conversion).
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(Vec::as_slice)
    }

    /// Token id for a raw byte sequence, if present.
    pub fn token_id(&self, bytes: &[u8]) -> Option<u32> {
        self.lookup.get(bytes).copied()
    }

    pub fn merge_count(&self) -> usize {
        self.merge_ranks.len()
    }

    pub fn mapped_char(&self, b: u8) -> char {
        self.byte_to_unicode[b as usize]
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn unk(&self) -> u32 {
        self.unk
    }

    /// Encode text into token ids (with BOS / EOS if the metadata requests it).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_special(text, self.add_bos, self.add_eos)
    }

    /// Encode without adding BOS / EOS. Useful for continuing a partial token.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_special(text, false, false)
    }

    fn encode_with_special(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if bos {
            ids.push(self.bos);
        }
        // Same normalization as `encode`, factored out for clarity.
        let normalized = match self.pre {
            PreTokenizer::SentencePiece => {
                let mut s = String::new();
                if !text.starts_with(' ') {
                    s.push(SPACE_MARK);
                }
                for c in text.chars() {
                    s.push(if c == ' ' { SPACE_MARK } else { c });
                }
                s
            }
            PreTokenizer::ByteBpe => text
                .as_bytes()
                .iter()
                .map(|b| self.byte_to_unicode[*b as usize])
                .collect(),
        };
        let symbols: Vec<Vec<u8>> = normalized
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                c.encode_utf8(&mut buf).as_bytes().to_vec()
            })
            .collect();
        ids.extend(self.bpe(symbols));
        if eos {
            ids.push(self.eos);
        }
        Ok(ids)
    }

    /// Merge symbols greedily, always taking the adjacent pair with the
    /// highest vocabulary score (this is how SentencePiece BPE and GPT-2 BPE
    /// both work).
    fn bpe(&self, symbols: Vec<Vec<u8>>) -> Vec<u32> {
        let mut ids: Vec<u32> = symbols
            .iter()
            .map(|s| *self.lookup.get(s).unwrap_or(&self.unk))
            .collect();

        loop {
            let mut best: Option<(usize, f32, usize, u32)> = None;
            for i in 0..ids.len().saturating_sub(1) {
                let mut merged = self.tokens[ids[i] as usize].clone();
                merged.extend_from_slice(&self.tokens[ids[i + 1] as usize]);
                if let Some(&new_id) = self.lookup.get(&merged) {
                    let rank = self.merge_ranks.get(&merged).copied().unwrap_or(usize::MAX);
                    let score = self.scores[new_id as usize];
                    let better = match best {
                        None => true,
                        Some((_, bs, br, _)) => {
                            if !self.merge_ranks.is_empty() {
                                rank < br
                            } else {
                                score > bs
                            }
                        }
                    };
                    if better {
                        best = Some((i, score, rank, new_id));
                    }
                }
            }
            match best {
                Some((i, _, _, id)) => {
                    ids[i] = id;
                    ids.remove(i + 1);
                }
                None => break,
            }
        }
        ids
    }

    /// Raw byte payload of a single token.
    pub fn decode_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let tok = self
            .tokens
            .get(id as usize)
            .ok_or(Error::InvalidToken(id))?;

        // LLaMA byte-fallback tokens are stored as `<0xXX>`.
        if tok.len() == 6 && tok[0] == b'<' && tok[1] == b'0' && tok[2] == b'x' && tok[5] == b'>' {
            if let (Some(hi), Some(lo)) = (hex(tok[3]), hex(tok[4])) {
                return Ok(vec![hi * 16 + lo]);
            }
        }

        match self.pre {
            PreTokenizer::SentencePiece => {
                // Replace `▁` with a normal space when the stored token uses it.
                if tok.starts_with(SPACE_MARK_BYTES) {
                    let mut out = vec![b' '];
                    out.extend_from_slice(&tok[SPACE_MARK_BYTES.len()..]);
                    Ok(out)
                } else {
                    Ok(tok.clone())
                }
            }
            PreTokenizer::ByteBpe => {
                let s = String::from_utf8_lossy(tok);
                let mut out = Vec::with_capacity(s.len());
                for c in s.chars() {
                    if let Some(b) = self.unicode_to_byte.get(&c) {
                        out.push(*b);
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
                Ok(out)
            }
        }
    }

    /// Decode a sequence of ids to a (lossy) UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Decode a chunk of streamed tokens into raw bytes.
    ///
    /// A single streamed token can end in the middle of a UTF-8 codepoint.
    /// Callers that print incrementally should keep the incomplete tail and
    /// prepend it to the next chunk (see `examples/generate.rs`).
    pub fn decode_stream(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_bytes(id)?);
        }
        Ok(bytes)
    }
}

#[inline]
fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// GPT-2's reversible byte <-> Unicode mapping.
fn byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut allowed: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(b'\xa1' as u32..=b'\xac' as u32)
        .chain(b'\xae' as u32..=b'\xff' as u32)
        .collect();
    let mut codes = allowed.clone();
    let mut n = 256u32;
    for b in 0u32..256 {
        if !allowed.contains(&b) {
            allowed.push(b);
            codes.push(n);
            n += 1;
        }
    }
    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::new();
    for i in 0..allowed.len() {
        let byte = allowed[i] as usize;
        let c = char::from_u32(codes[i]).unwrap_or('\u{fffd}');
        b2u[byte] = c;
        u2b.insert(c, byte as u8);
    }
    (b2u, u2b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_mapping_values() {
        let (b2u, _) = byte_unicode_tables();
        assert_eq!(b2u[79], 'O', "codes: {:?}", &b2u[70..100]);
        assert_eq!(b2u[32], '\u{120}'); // space -> Ġ
        assert_eq!(b2u[110], 'n');
    }

    #[test]
    fn byte_mapping_is_reversible() {
        let (b2u, u2b) = byte_unicode_tables();
        for b in 0u8..=255 {
            assert_eq!(u2b[&b2u[b as usize]], b);
        }
    }

    #[test]
    fn bpe_merges_by_score() {
        // Synthetic vocab: "a"=0, "b"=1, "ab"=2 (higher score)
        let mut t = Tokenizer {
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
            scores: vec![0.0, 0.0, 10.0],
            lookup: HashMap::new(),
            merge_ranks: HashMap::new(),
            bos: 0,
            eos: 1,
            unk: 0,
            add_bos: false,
            add_eos: false,
            pre: PreTokenizer::SentencePiece,
            byte_to_unicode: byte_unicode_tables().0,
            unicode_to_byte: byte_unicode_tables().1,
        };
        for (i, tok) in t.tokens.clone().iter().enumerate() {
            t.lookup.insert(tok.clone(), i as u32);
        }
        assert_eq!(t.bpe(vec![b"a".to_vec(), b"b".to_vec()]), vec![2]);
    }
}
