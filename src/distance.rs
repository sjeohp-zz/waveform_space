extern crate rustfft;
extern crate num;
extern crate stream_dct;
extern crate rand;

use rustfft::FFT;
use num::complex::Complex;
use std::iter::Peekable;
use std::f32;

const LOWER_FREQ: 		f32 = 20.0;
const UPPER_FREQ: 		f32 = 8000.0;
const FRAME_DURATION: 	f32 = 0.025;
const STEP_DURATION: 	f32 = 0.01;

pub fn distance_between(samples_a: &[f32], samples_b: &[f32], samplerate: f32) -> f32
{
	let frames_a = frames(samples_a, samplerate);
	let frames_b = frames(samples_b, samplerate);
	let mut sum = 0f32;
	for i in 0..frames_a.len()
	{
		let mfcc_a = mfcc(&frames_a[i], samplerate);
		let mfcc_b = mfcc(&frames_b[i], samplerate);
		for j in 0..mfcc_a.len()
		{
			sum += (mfcc_a[j] - mfcc_b[j]).powi(2);
		}
	}
	sum
}

fn frames(samples: &[f32], samplerate: f32) -> Vec<Vec<f32>>
{
	let nsamples_frame = (FRAME_DURATION*samplerate) as usize;
	let nsamples_step = (STEP_DURATION*samplerate) as usize;

	let mut frames = vec![];
	let mut start = 0;
	while start + nsamples_frame < samples.len()
	{
		frames.push(samples[start..start+nsamples_frame].to_vec());
		start += nsamples_step;
	}
	let mut last = samples[start..].to_vec();
	last.resize(nsamples_frame, 0f32);
	frames.push(last);
	frames
}

fn mfcc(frame: &[f32], samplerate: f32) -> Vec<f32>
{
	let fft_size = pow2_roundup(frame.len());
	let filter_size = fft_size/2+1;
	let filterbank = filterbank(26, fft_size, filter_size, samplerate);
	let spectrum = spectrum(frame).into_iter().take(filter_size).collect::<Vec<f32>>();
	let energies: Vec<f64> = energies(filterbank, spectrum).iter().map(|x| *x as f64).collect();
	stream_dct::DCT1D::new(&energies[..]).take(12).map(|x| x as f32).collect::<Vec<f32>>()
}

fn pow2_roundup(mut x: usize) -> usize
{
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

fn freq_to_mel(f: f32) -> f32
{
	1125.0*(1.0 + f/700.0).ln()
}

fn mel_to_freq(m: f32) -> f32
{
	((m/1125.0).exp() - 1.0) * 700.0
}

fn freq_to_bucket(f: f32, fft_size: usize, samplerate: f32) -> i32
{
	((fft_size as f32 + 1.0)*f/samplerate).floor() as i32
}

fn filter(filter_size: usize, mbot: f32, mzen: f32, mtop: f32) -> Vec<f32>
{
	let mut filter = vec![0f32; filter_size];
	for kx in 0..filter.len()
	{
		let k = kx as f32;
		if k < mbot { continue; }
		else if k > mbot && k < mzen { filter[kx] = (k-mbot)/(mzen-mbot); }
		else if k > mzen && k < mtop { filter[kx] = (mtop-k)/(mtop-mzen); }
		else { continue; }
	}
	filter
}

fn filterbank(nfilters: usize, fft_size: usize, filter_size: usize, samplerate: f32) -> Vec<Vec<f32>> 
{
	let mut filterbank = [0; 26];

	let mel_lower = freq_to_mel(LOWER_FREQ);
	let mel_upper = freq_to_mel(UPPER_FREQ);
	let mel_step = (mel_upper - mel_lower) / 11.0;

	let freq = (0..nfilters+2).map(move |x| mel_to_freq(mel_lower+(x as f32)*mel_step));
	let buckets: Vec<i32> = freq.map(move |x| freq_to_bucket(x, fft_size, samplerate)).collect();

	let mut filterbank = vec![];
	for m in 1..nfilters+1
	{
		filterbank.push(filter(filter_size, buckets[m-1] as f32, buckets[m] as f32, buckets[m+1] as f32));
	}
	filterbank
}

fn spectrum(frame: &[f32]) -> Vec<f32>
{
	let mut fft = FFT::new(frame.len(), false);
	let signal = frame.iter().map(|x| Complex::new(*x, 0f32)).collect::<Vec<_>>();
	let mut spectrum = signal.clone();
	fft.process(&signal[..], &mut spectrum[..]);
	spectrum.iter().map(|x| x.norm_sqr()).collect::<Vec<f32>>()
}

fn energies(filterbank: Vec<Vec<f32>>, spectrum: Vec<f32>) -> Vec<f32>
{
	let mut ener = Vec::with_capacity(filterbank.len());
	for filter in filterbank.iter()
	{
		let val: f32 = filter.into_iter().zip(spectrum.clone().into_iter()).map(|(x, y)| x*y).sum();
		ener.push(if val > 0.0 { val.ln() } else { 0.0 });
	}
	ener
}