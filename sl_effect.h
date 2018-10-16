/* -------- C++ Sound Library for Signal Processing --------
This library needs Eigen.
http://eigen.tuxfamily.org/index.php?title=Main_Page

Copyright (c) Hayato OHYA 2013 All Rights Reserved.
E-mail : hayato@mlab.phys.waseda.ac.jp

reference
[1] Naofumi AOKI, C言語ではじめる音のプログラミング, 2008 (in Japanese)
--------------------------------------------------------------*/
#include <Eigen/Core>
#include <iostream>

namespace sl
{
	Wav offVocal(Wav);
	Wav delay(Wav wavData, double dampingRate = 0.5, double delayTime = 0.3, int repeatTime = 3);

	/* Function */
	Wav offVocal(Wav wavData)
	{
		/*Get vocal off from a two channel song data*/
		// INPUT : Two channel WAVE data
		// OUTPUT :One channel WAVE data 
		
		Wav output;
		Eigen::MatrixXd subtractMat;
		if (wavData.channel != 2){
			cout << "ERROR : The number of channel is not 2" << endl;
			return wavData;
		}
		subtractMat = wavData.data.col(0) - wavData.data.col(1);
		// assignment
		output.bits = wavData.bits;
		output.channel = 1;
		output.data = subtractMat;
		output.fs = wavData.fs;
		output.length = wavData.length;

		return output;
	}
	Wav delay(Wav wavData, double dampingRate, double delayTime, int repeatTime)
	{
		Wav output;
		int m;
		output = wavData;
		output.data = Eigen::MatrixXd::Zero(wavData.data.rows(), wavData.data.cols());

		delayTime *= wavData.fs;
		
		if (wavData.channel == 1){
			for (int n=0; n<wavData.length; n++){
				output.data(n,0) = wavData.data(n,0); // current data
				for (int i=1; i<=repeatTime; i++){
					m = (int) ((double)n - (double) i*delayTime);
					if (m>=0){
						output.data(m,0) += pow(dampingRate, (double) i) * wavData.data(m,0);
					}
				}
			}
		}else{
			for (int n=0; n<wavData.length; n++){
				output.data(n,0) = wavData.data(n,0); // current data
				output.data(n,1) = wavData.data(n,1);
				for (int i=1; i<=repeatTime; i++){
					m = (int) ((double)n - (double) i*delayTime);
					if (m>=0){
						output.data(m,0) += pow(dampingRate, (double) i) * wavData.data(m,0);
						output.data(m,1) += pow(dampingRate, (double) i) * wavData.data(m,1);
					}
				}
			}
		}

		return output;
	}
}