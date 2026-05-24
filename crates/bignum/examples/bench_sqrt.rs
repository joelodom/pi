use bignum::Float;
use std::time::Instant;

fn main() {
    for prec in [3_320_000_u64, 332_000, 100_000, 33_000, 10_000] {
        let mut f = Float::with_val_64(prec, 10005_u32);
        let t = Instant::now();
        f.sqrt_mut();
        let elapsed = t.elapsed();
        println!("sqrt @ {prec:>8} bits: {elapsed:?}");
    }
}
