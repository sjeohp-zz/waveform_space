extern crate rustfft;
extern crate num;
extern crate stream_dct;
extern crate rand;

pub mod distance;

use distance::*;

#[cfg(test)]
mod unit_tests {
	use super::*;

	use rand::{thread_rng, Rng};
	use std::f32;

	#[test]
	fn test_wave_comparison()
	{
		for z in 0..10
		{
			// if z % 50 == 0 { println!("{}", z); }

			let samplerate = 16000.0;

			let mut rng_a = thread_rng();
			let mut rng_b = thread_rng();
			let mut rng_c = thread_rng();

			let rng_factor_a = 0.0;
			let rng_factor_b = rng_b.gen::<f32>();
			let rng_factor_c = rng_c.gen::<f32>();

			let wave_a: Vec<f32> = (0u64..samplerate as u64).map(move |t| t as f32 * 440.0 * 2.0*f32::consts::PI / samplerate).map(move |t| (t.sin() + rng_factor_a*(rng_a.gen::<f32>()*2.0-1.0)) * 0.1).collect();
			let wave_b: Vec<f32> = (0u64..samplerate as u64).map(move |t| t as f32 * 440.0 * 2.0*f32::consts::PI / samplerate).map(move |t| (t.sin() + rng_factor_b*(rng_b.gen::<f32>()*2.0-1.0)) * 0.1).collect();
			let wave_c: Vec<f32> = (0u64..samplerate as u64).map(move |t| t as f32 * 440.0 * 2.0*f32::consts::PI / samplerate).map(move |t| (t.sin() + rng_factor_c*(rng_c.gen::<f32>()*2.0-1.0)) * 0.1).collect();

			let dist_ab = distance_between(
				&wave_a, 
				&wave_b,
				samplerate);
			let dist_ac = distance_between(
				&wave_a, 
				&wave_c,
				samplerate);
			let dist_bc = distance_between(
				&wave_b, 
				&wave_c,
				samplerate);

			let pass = ((rng_factor_a - rng_factor_b).abs() > (rng_factor_a - rng_factor_c).abs()) == (dist_ab.abs() > dist_ac.abs());

			if !pass 
			{
				println!("--rng factors");
				println!("{:?}", rng_factor_a);
				println!("{:?}", rng_factor_b);
				println!("{:?}", rng_factor_c);
				println!("--rng deltas");
				println!("{:?}", (rng_factor_a - rng_factor_b).abs());
				println!("{:?}", (rng_factor_a - rng_factor_c).abs());
				println!("{:?}", (rng_factor_b - rng_factor_c).abs());
				println!("--sum deltas");
				println!("{:?}", dist_ab.abs());
				println!("{:?}", dist_ac.abs());
				println!("{:?}", dist_bc.abs());
			}
			assert!(pass);
		}		
	}
}