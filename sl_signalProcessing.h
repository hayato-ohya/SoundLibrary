/* -------- C++ Sound Library for Signal Processing --------
This library needs Eigen.
http://eigen.tuxfamily.org/index.php?title=Main_Page

Copyright (c) Hayato OHYA 2013 All Rights Reserved.
E-mail : hayato@mlab.phys.waseda.ac.jp

reference
[1] Naofumi AOKI, C言語ではじめる音のプログラミング, 2008 (in Japanese)
[2] http://www.kurims.kyoto-u.ac.jp/~ooura/index-j.html
--------------------------------------------------------------*/
#include <Eigen/Core>
#include <iostream>
#include "fftsg_ooura.h" // http://www.kurims.kyoto-u.ac.jp/~ooura/index-j.html

namespace sl
{
	// Definition-------------------------------------------------------------------------------
	double PI = 3.1415926535;
	int WINDOWSIZE = 4096; // default window size
	int repTimes = 30; // The number of repetition times that is used for "sawtoothSin"
	// -----------------------------------------------------------------------------------------

	// window function
	Eigen::VectorXd hanningWindow(int windowSize = WINDOWSIZE);
	Eigen::VectorXd hammingWindow(int windowSize = WINDOWSIZE);

	// fft
	Fft fft(Eigen::VectorXd, int isign=-1);
	Fft fft(Fft, int isign=-1);

	// spectrogram
	Spec spectrogram(Wav, int fftPoints = 4096);

	// generate fast or slow Wave data
	Wav changeWavSpeed(Wav wavData, double rate);
	Wav generateFastPlayWav(Wav wavData, double rate);
	Wav generateSlowPlayWav(Wav wavData, double rate);

	// resampling
	Wav resampling(Wav wavData, double rate);

	// pitch shift
	Wav pitchShifter(Wav wavData, double rate);
	Wav pitchShifter(Wav wavData, int key);

	// Generate sine curve (1 channel, default 16bit, 44100Hz)
	Wav normalSin(int f0, int length, double A); // Normal sine curve
	Wav normalSin(int f0, int length, double A, int variable); // Normal sine curve
	Wav normalSin(int f0, int length, double A, int bits, int fs); // Normal sine curve
	Wav normalSin(int f0, double sec, double A); // Normal sine curve
	Wav normalSin(int f0, double sec, double A, int variable); // Normal sine curve
	Wav normalSin(int f0, double sec, double A, int bits, int fs); // Normal sine curve
	// Saw-tooth sine curve (Sine curves are added 30 times)
	Wav sawtoothSin(int f0, int length, double A); // Saw-tooth sine curve
	Wav sawtoothSin(int f0, int length, double A, int variable); // Saw-tooth sine curve
	Wav sawtoothSin(int f0, int length, double A, int bits, int fs); // Saw-tooth sine curve
	Wav sawtoothSin(int f0, double sec, double A); // Saw-tooth sine curve
	Wav sawtoothSin(int f0, double sec, double A, int variable); // Saw-tooth sine curve
	Wav sawtoothSin(int f0, double sec, double A, int bits, int fs); // Saw-tooth sine curve

	Eigen::VectorXd hanningWindow(int windowSize)
	{
		/* Hanning Window */
		int i;
		Eigen::VectorXd output = Eigen::VectorXd::Zero(windowSize);

		if (windowSize % 2 == 0) // windowSize is even
		{
			for(i=0; i<windowSize; i++){
				output(i) = 0.5 - 0.5 * cos(2.0 * PI * i / windowSize);
			}
		}else{ // windowSize is odd
			for(i=0; i<windowSize; i++){
				output(i) = 0.5 - 0.5 * cos(2.0 * PI * (i + 0.5) / windowSize);
			}
		}

		return output;
	}

	Eigen::VectorXd hammingWindow(int windowSize)
	{
		/* Hamming Window */
		int i;
		Eigen::VectorXd output = Eigen::VectorXd::Zero(windowSize);

		if (windowSize % 2 == 0) // windowSize is even
		{
			for(i=0; i<windowSize; i++){
				output(i) = 0.54 - 0.46 * cos(2.0 * PI * i / windowSize);
			}
		}else{ // windowSize is odd
			for(i=0; i<windowSize; i++){
				output(i) = 0.54 - 0.46 * cos(2.0 * PI * (i + 0.5) / windowSize);
			}
		}

		return output;
	}

	Fft fft(Eigen::VectorXd input, int isign)
	{
		/* FFT function */
		// reference
		// [1] http://www.kurims.kyoto-u.ac.jp/~ooura/index-j.html
		// [2] http://geisterchor.blogspot.jp/2011/04/fft_16.html
		// INPUT : Vector data
		// isign : -1(dft) or 1(idft)
		// OUTPUT : Fft structure which includes a real part and a imaginary part vectors
		Fft fftData;
		int length, counter, temp, i;
		Eigen::VectorXd x_real, x_imag;

		// Ooura's cdft valiable
		int n; // double number of FFT points
		double *a;
		int *ip;
		double *w;

		/* frame size decision and initialization */
		counter = 1;
		while(1){
			temp = (int) pow(2,counter);
			if (temp == input.size()){
				length = input.size();
				// initialzation
				x_real = input;
				x_imag = Eigen::VectorXd::Zero(length);
				break;
			}else if(temp > input.size()){ // if length of input vector is not power-of-two
				length = temp;
				// initialzation
				x_real = Eigen::VectorXd::Zero(length);
				for (int i=0; i< input.size(); i++){
					x_real(i) = input(i);
				}
				x_imag = Eigen::VectorXd::Zero(length);
				break;
			}
			counter++;
		}

		/* Ooura's cdft function */
		n = x_real.size() + x_imag.size();
		// memory allocation
		a = (double *) malloc(sizeof(double) * n);
		ip = (int *) malloc(sizeof(int) * (2+sqrt(n)));
		w = (double *) malloc(sizeof(double) * (n / 2));

		// assignment of input array
		for (i=0; i<x_real.size(); i++){
			a[i*2] = x_real(i);
			a[i*2 + 1] = x_imag(i);
		}

		// FFT
		ip[0] = 0.0;
		cdft(n, isign, a, ip, w);

		// assignment of Eigen::MatrixXd
		for (i=0; i<x_real.size(); i++){
			x_real(i) = a[i*2];
			x_imag(i) = a[i*2 + 1];
		}

		fftData.real = x_real / sqrt(x_real.size());
		fftData.imag = x_imag / sqrt(x_imag.size());

		free(w);
		free(ip);
		free(a);

		return fftData;
	}

	Fft fft(Fft input, int isign)
	{
		/* FFT function */
		// INPUT : Fft structure which includes a real part and a imaginary part vectors
		// OUTPUT : Fft structure which includes a real part and a imaginary part vectors
		Fft fftData;
		int length, counter, temp, i;
		Eigen::VectorXd x_real, x_imag;

		// Ooura's cdft valiable
		int n; // double number of FFT points
		double *a;
		int *ip;
		double *w;

		/* frame size decision and initialization */
		counter = 1;
		while(1){
			temp = (int) pow(2,counter);
			if (temp == input.real.size()){
				length = input.real.size();
				// initialzation
				x_real = input.real;
				x_imag = input.imag;
				break;
			}else if(temp > input.real.size()){ // if length of input vector is not power-of-two
				length = temp;
				// initialzation
				x_real = Eigen::VectorXd::Zero(length);
				x_imag = Eigen::VectorXd::Zero(length);
				for (int i=0; i< input.real.size(); i++){
					x_real(i) = input.real(i);
				}
				for (int i=0; i< input.real.size(); i++){
					x_imag(i) = input.imag(i);
				}
				break;
			}
			counter++;
		}

		/* Ooura's cdft function */
		n = x_real.size() + x_imag.size();
		// memory allocation
		a = (double *) malloc(sizeof(double) * n);
		ip = (int *) malloc(sizeof(int) * (2+sqrt(n)));
		w = (double *) malloc(sizeof(double) * (n / 2));

		// assignment of input array
		for (i=0; i<x_real.size(); i++){
			a[i*2] = x_real(i);
			a[i*2 + 1] = x_imag(i);
		}

		// FFT
		ip[0] = 0.0;
		cdft(n, isign, a, ip, w);

		// assignment of Eigen::MatrixXd
		for (i=0; i<x_real.size(); i++){
			x_real(i) = a[i*2];
			x_imag(i) = a[i*2 + 1];
		}

		fftData.real = x_real / sqrt(x_real.size());
		fftData.imag = x_imag / sqrt(x_imag.size());

		free(w);
		free(ip);
		free(a);

		return fftData;
	}

	Spec spectrogram(Wav wavData, int fftPoints)
	{
		/* Calculate spectrogram like MATLAB */
		// INPUT : Wave file structure
		// OUTPUT : spectrogram structure
		int rows, remainder;
		Eigen::MatrixXd specMat; // spectrogram matrix (row:time, col:power spectrum of frequency)
		Fft fftData;
		Spec output;
		output.fftPoints = fftPoints;
		output.fs = wavData.fs;

		// monauralize
		if (wavData.channel != 1){
			wavData = stereo2mono(wavData);
		}

		// calculate the number of rows
		remainder = wavData.length % fftPoints;
		if (remainder == 0){
			rows = wavData.length / fftPoints;
		}else{
			rows = wavData.length / fftPoints + 1;
		}

		// FFT
		specMat = Eigen::MatrixXd::Zero(rows,fftPoints);
		if (remainder == 0){
			for (int i=0; i<rows; i++){
				fftData = fft(wavData.data.col(0).segment(i*fftPoints, fftPoints).cwiseProduct(hanningWindow(fftPoints)));
				specMat.row(i) = iPow(fftData.real, fftData.imag);
			}
		}else{
			for (int i=0; i<rows-1; i++){
				fftData = fft(wavData.data.col(0).segment(i*fftPoints, fftPoints).cwiseProduct(hanningWindow(fftPoints)));
				specMat.row(i) = iPow(fftData.real, fftData.imag);
			}
			fftData = fft(wavData.data.col(0).segment(wavData.length-fftPoints, fftPoints).cwiseProduct(hanningWindow(fftPoints)));
			specMat.row(rows-1) = iPow(fftData.real, fftData.imag);
		}
		output.data = specMat;

		return output;
	}

	Wav changeWavSpeed(Wav wavData, double rate)
	{
		Wav output;
		if (rate > 1.0){
			output = generateFastPlayWav(wavData, rate);
		}else if(0.5 <= rate && rate < 1.0){
			output = generateSlowPlayWav(wavData, rate);
		}else if(rate == 1.0){
			output = wavData;
		}else{
			cout << "ERROR : The value of rate must be more than 0.5.\n" << endl;
			output = wavData;
		}
		return output;
	}

	Wav generateFastPlayWav(Wav wavData, double rate)
	{
		/* generate a wave data for playing at fast speed */
		// INPUT : Wave file structure, speed rate (rate > 1.0)
		Wav output;
		output.bits = wavData.bits;
		output.channel = wavData.channel;
		output.fs = wavData.fs;
		output.length = (int) (wavData.length / rate) + 1;
		output.data = Eigen::MatrixXd::Zero(output.length, output.channel);

		// variable for auto-correlation function
		int templateSize = (int) (output.fs * 0.01); // 10ms
		int pmin = (int) (output.fs * 0.005); // 5ms
		int pmax = (int) (output.fs * 0.02); // 20ms

		Eigen::VectorXd x = Eigen::VectorXd::Zero(templateSize);
		Eigen::VectorXd y = Eigen::VectorXd::Zero(templateSize);
		Eigen::VectorXd r = Eigen::VectorXd::Zero(pmax+1);

		int offset0 = 0;
		int offset1 = 0;
		int q;

		// mono
		Wav mono_wav;
		Eigen::VectorXd mono;
		if(wavData.channel == 2){
			mono_wav = stereo2mono(wavData);
			mono = mono_wav.data;
		}else{
			mono = wavData.data;
		}

		while (offset0 + pmax*2 < wavData.length){
			for (int n=0; n<templateSize; n++){
				x(n) = mono(offset0+n); // original soud data
			}

			double max_r = 0.0;
			int p = pmin;
			for (int m=pmin; m<=pmax; m++){
				for (int n=0; n<templateSize; n++){
					y(n) = mono(offset0+m+n);
				}
				r(m) = 0.0;
				for (int n=0; n<templateSize; n++){
					r(m) += x(n,0) + y(n,0); // auto-correlation function
				}
				if (r(m) > max_r){
					max_r = r(m); // peak of auto-correlation function
					p = m; // fundamental period
				}
			}

			for (int n=0; n<p; n++){
				output.data.row(offset1+n) = wavData.data.row(offset0+n) * (p-n) / p; // monotonically decreasing
				output.data.row(offset1+n) += wavData.data.row(offset0+p+n) * n / p; // monotonically increasing
			}

			q = (int)(p / (rate - 1.0) + 0.5);
			for (int n=p; n<q; n++){
				if(offset0+p+n >= wavData.length){
					break;
				}
				output.data.row(offset1+n) = wavData.data.row(offset0 + p + n);
			}
			offset0 += p+q;
			offset1 += q;
		}

		return output;
	}

	Wav generateSlowPlayWav(Wav wavData, double rate)
	{
		/* generate a wave data for playing at slow speed */
		// INPUT : Wave file structure, speed rate (0.5 <= rate < 1.0)
		Wav output;
		output.bits = wavData.bits;
		output.channel = wavData.channel;
		output.fs = wavData.fs;
		output.length = (int) (wavData.length / rate) + 1;
		output.data = Eigen::MatrixXd::Zero(output.length, output.channel);

		// variable for auto-correlation function
		int templateSize = (int) (output.fs * 0.01); // 10ms
		int pmin = (int) (output.fs * 0.005); // 5ms
		int pmax = (int) (output.fs * 0.02); // 20ms

		Eigen::VectorXd x = Eigen::VectorXd::Zero(templateSize);
		Eigen::VectorXd y = Eigen::VectorXd::Zero(templateSize);
		Eigen::VectorXd r = Eigen::VectorXd::Zero(pmax+1);

		int offset0 = 0;
		int offset1 = 0;
		int q;

		// mono
		Wav mono_wav;
		Eigen::VectorXd mono;
		if(wavData.channel == 2){
			mono_wav = stereo2mono(wavData);
			mono = mono_wav.data;
		}else{
			mono = wavData.data;
		}

		while (offset0 + pmax*2 < wavData.length){
			for (int n=0; n<templateSize; n++){
				x(n) = mono(offset0+n); // original soud data
			}
			
			double max_r = 0.0;
			int p = pmin;
			for (int m=pmin; m<=pmax; m++){
				for (int n=0; n<templateSize; n++){
					y(n) = mono(offset0+m+n);
				}
				r(m) = 0.0;
				for (int n=0; n<templateSize; n++){
					r(m) += x(n) + y(n); // auto-correlation function
				}
				if (r(m) > max_r){
					max_r = r(m); // peak of auto-correlation function
					p = m; // fundamental period
				}
			}
			
			for (int n=0; n<p; n++){
				output.data.row(offset1+n) = wavData.data.row(offset0+n);
			}
			for (int n=0; n<p; n++){
				output.data.row(offset1+p+n) = wavData.data.row(offset0+p+n) * (p-n) / p; // monotonically decreasing
				output.data.row(offset1+p+n) += wavData.data.row(offset0+n) * n / p; // monotonically increasing
			}
			
			q = (int)(p * rate / (1.0 - rate) + 0.5);
			for (int n=p; n<q; n++){
				if(offset0+n >= wavData.length){
					break;
				}
				if (output.length == offset1+p+n){
					break;
				}
				output.data.row(offset1+p+n) = wavData.data.row(offset0+n);
			}
			
			offset0 += q;
			offset1 += p + q;
			
		}

		return output;
	}

	Wav resampling(Wav wavData, double rate)
	{
		double t;
		int offset;
		Wav output;
		output.fs = wavData.fs;
		output.bits = wavData.bits;
		output.length = (int) (wavData.length / rate);
		output.channel = wavData.channel;
		output.data = Eigen::MatrixXd::Zero(output.length, wavData.channel);

		int J = 24;

		for (int n=0; n<output.length; n++){
			t = rate * n;
			offset = (int) t;
			for(int m=offset-J/2; m<=offset+J/2; m++){
				if(m >= 0 && m < wavData.length){
					output.data.row(n) += wavData.data.row(m) * sinc(PI * (t-m));
				}
			}
		}

		return output;
	}

	Wav pitchShifter(Wav wavData, double rate)
	{
		Wav wavData_resampled, output;
		wavData_resampled = resampling(wavData, rate);
		output = changeWavSpeed(wavData_resampled, 1/rate);
		return output;
	}

	Wav normalSin(int f0, int length, double A)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		int i;
		int bits = 16;
		int fs = 44100;
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav normalSin(int f0, int length, double A, int variable)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// variable : quantization precision (bits) or sampling rate (fs)
		int i, fs, bits;
		Wav output;
		Eigen::MatrixXd sinData;

		// discriminate variable
		if (variable == 8 || variable == 16 || variable == 24 || variable == 32){
			bits = variable;
			fs = 44100;
		}else{
			fs = variable;
			bits = 16;
		}

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav normalSin(int f0, int length, double A, int bits, int fs)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// bits : Quantization precision
		// fs : Sampling rate
		int i;
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav normalSin(int f0, double sec, double A)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		int i;
		int bits = 16;
		int fs = 44100;
		int length = (int) floor(sec * fs);
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav normalSin(int f0, double sec, double A, int variable)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// variable : quantization precision (bits) or sampling rate (fs)
		int i, fs, bits;
		Wav output;
		Eigen::MatrixXd sinData;

		// discriminate variable
		if (variable == 8 || variable == 16 || variable == 24 || variable == 32){
			bits = variable;
			fs = 44100;
		}else{
			fs = variable;
			bits = 16;
		}
		int length = (int) floor(sec * fs);

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav normalSin(int f0, double sec, double A, int bits, int fs)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// bits : Quantization precision
		// fs : Sampling rate
		int i;
		Wav output;
		Eigen::MatrixXd sinData;
		int length = (int) floor(sec * fs);

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			sinData(i,0) = A * sin(2 * PI  * f0 * i / fs);
		}
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, int length, double A)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		int i, j;
		int bits = 16;
		int fs = 44100;
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, int length, double A, int variable)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// variable : quantization precision (bits) or sampling rate (fs)
		int i, j, fs, bits;
		Wav output;
		Eigen::MatrixXd sinData;

		// discriminate variable
		if (variable == 8 || variable == 16 || variable == 24 || variable == 32){
			bits = variable;
			fs = 44100;
		}else{
			fs = variable;
			bits = 16;
		}

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, int length, double A, int bits, int fs)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// bits : Quantization precision
		// fs : Sampling rate
		int i, j;
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, double sec, double A)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		int i, j;
		int bits = 16;
		int fs = 44100;
		int length = (int) floor(sec * fs);
		Wav output;
		Eigen::MatrixXd sinData;

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, double sec, double A, int variable)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// variable : quantization precision (bits) or sampling rate (fs)
		int i, j, fs, bits;
		Wav output;
		Eigen::MatrixXd sinData;

		// discriminate variable
		if (variable == 8 || variable == 16 || variable == 24 || variable == 32){
			bits = variable;
			fs = 44100;
		}else{
			fs = variable;
			bits = 16;
		}
		int length = (int) floor(sec * fs);

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}

	Wav sawtoothSin(int f0, double sec, double A, int bits, int fs)
	{
		// INPUT:
		// f0 : Fundamental frequency
		// length : WAVE data length
		// A : Amplitude
		// bits : Quantization precision
		// fs : Sampling rate
		int i, j;
		Wav output;
		Eigen::MatrixXd sinData;
		int length = (int) floor(sec * fs);

		// assignment
		output.bits = bits;
		output.channel = 1;
		output.fs = fs;
		output.length = length;
		sinData = Eigen::MatrixXd::Zero(length, 1);
		for (i=0; i<length; i++){
			for (j=1; j<=repTimes; j++){
				sinData(i,0) += (A / j) * sin(2 * PI  *( f0 * j) * i / fs);
			}
		}
		sinData = sinData / sinData.col(0).maxCoeff() * A;
		output.data = sinData;
		return output;
	}
}