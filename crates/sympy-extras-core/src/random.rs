//! A small deterministic pseudo-random generator (SplitMix64) for the
//! randomised algorithms (the equal-degree splitting of Cantor and
//! Zassenhaus).
//!
//! The Python package uses `random.Random` (the Mersenne Twister) with a
//! fixed seed. The results of the algorithms do not depend on the random
//! choices (the factor lists are sorted), only the functions that return
//! random objects (`gf_random`) differ from Python's.

/// SplitMix64 (Steele, Lea, Flood, *Fast splittable pseudorandom number
/// generators*, OOPSLA 2014).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z: u64 = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..n` (`n > 0`; Python's `randrange(n)`), by
    /// rejection so that there is no bias.
    pub fn randrange(&mut self, n: u64) -> u64 {
        assert!(n > 0, "empty range for randrange");
        let zone: u64 = u64::MAX - (u64::MAX % n);
        loop {
            let v: u64 = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
}

// ---------------------------------------------------------------------
// Python's random.Random: the Mersenne Twister and its seeding, where a
// port must reproduce Python's choices exactly

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// The SHA-512 digest (FIPS 180-4), for Python's seeding by a string.
pub fn sha512(message: &[u8]) -> [u8; 64] {
    let mut h: [u64; 8] = [
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    ];
    let mut data: Vec<u8> = message.to_vec();
    data.push(0x80);
    while data.len() % 128 != 112 {
        data.push(0);
    }
    data.extend_from_slice(&((message.len() as u128) * 8).to_be_bytes());
    for block in data.chunks(128) {
        let mut w: [u64; 80] = [0; 80];
        for (t, word) in block.chunks(8).enumerate() {
            w[t] = u64::from_be_bytes(word.try_into().expect("8 bytes"));
        }
        for t in 16..80 {
            let s0: u64 = w[t - 15].rotate_right(1) ^ w[t - 15].rotate_right(8) ^ (w[t - 15] >> 7);
            let s1: u64 = w[t - 2].rotate_right(19) ^ w[t - 2].rotate_right(61) ^ (w[t - 2] >> 6);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let mut v: [u64; 8] = h;
        for t in 0..80 {
            let s1: u64 = v[4].rotate_right(14) ^ v[4].rotate_right(18) ^ v[4].rotate_right(41);
            let ch: u64 = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1: u64 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[t])
                .wrapping_add(w[t]);
            let s0: u64 = v[0].rotate_right(28) ^ v[0].rotate_right(34) ^ v[0].rotate_right(39);
            let maj: u64 = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2: u64 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out: [u8; 64] = [0; 64];
    for (i, word) in h.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&word.to_be_bytes());
    }
    out
}

const MT_N: usize = 624;
const MT_M: usize = 397;

/// Python's `random.Random` (`PyRandom`): the Mersenne Twister MT19937
/// with CPython's seeding (`init_by_array` of the 32-bit words of the
/// seed) and its integer methods.
#[derive(Debug, Clone)]
pub struct PyRandom {
    mt: Vec<u32>,
    index: usize,
}

impl PyRandom {
    fn init_genrand(s: u32) -> Vec<u32> {
        let mut mt: Vec<u32> = vec![0; MT_N];
        mt[0] = s;
        for i in 1..MT_N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        mt
    }

    fn from_key(key: &[u32]) -> PyRandom {
        let mut mt: Vec<u32> = PyRandom::init_genrand(19650218);
        let (mut i, mut j): (usize, usize) = (1, 0);
        for _ in 0..MT_N.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..MT_N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
        }
        mt[0] = 0x80000000;
        PyRandom { mt, index: MT_N }
    }

    /// `random.Random(n)` for an integer seed (its absolute value).
    pub fn from_int(n: &crate::numbers::Int) -> PyRandom {
        use dashu_base::UnsignedAbs;
        let bytes: Vec<u8> = n.clone().unsigned_abs().to_le_bytes().to_vec();
        let mut key: Vec<u32> = bytes
            .chunks(4)
            .map(|c| {
                let mut word: [u8; 4] = [0; 4];
                word[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(word)
            })
            .collect();
        while key.len() > 1 && key[key.len() - 1] == 0 {
            key.pop();
        }
        if key.is_empty() {
            key.push(0);
        }
        PyRandom::from_key(&key)
    }

    /// `random.Random(s)` for a string seed (version 2: the bytes and their
    /// SHA-512 digest as one big-endian integer).
    pub fn from_string(s: &str) -> PyRandom {
        let mut bytes: Vec<u8> = s.as_bytes().to_vec();
        bytes.extend_from_slice(&sha512(s.as_bytes()));
        let n: crate::numbers::Int = crate::numbers::Int::from(dashu_int::UBig::from_be_bytes(&bytes));
        PyRandom::from_int(&n)
    }

    fn generate(&mut self) {
        for k in 0..MT_N {
            let y: u32 = (self.mt[k] & 0x80000000) | (self.mt[(k + 1) % MT_N] & 0x7fffffff);
            let mut next: u32 = self.mt[(k + MT_M) % MT_N] ^ (y >> 1);
            if y & 1 == 1 {
                next ^= 0x9908b0df;
            }
            self.mt[k] = next;
        }
        self.index = 0;
    }

    /// The next 32-bit output (`genrand_uint32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            self.generate();
        }
        let mut y: u32 = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for `0 < k <= 64`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k > 0 && k <= 64, "getrandbits of 1 to 64 bits");
        if k <= 32 {
            return u64::from(self.next_u32() >> (32 - k));
        }
        let low: u64 = u64::from(self.next_u32());
        let high: u64 = u64::from(self.next_u32() >> (64 - k));
        low | (high << 32)
    }

    /// `_randbelow(n)` by `getrandbits` and rejection, `n > 0`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k: u32 = 64 - n.leading_zeros();
        loop {
            let r: u64 = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `randint(a, b)`, both included.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        a + self.randbelow((b - a + 1) as u64) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        for _ in 0..1000 {
            let x = a.randrange(7);
            assert_eq!(x, b.randrange(7));
            assert!(x < 7);
        }
    }

    #[test]
    fn sha512_vectors() {
        let hex = |d: [u8; 64]| d.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        assert!(hex(super::sha512(b"abc")).starts_with("ddaf35a193617aba"));
        assert!(hex(super::sha512(b"")).starts_with("cf83e1357eefb8bd"));
    }
}
