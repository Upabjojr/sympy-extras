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
}
