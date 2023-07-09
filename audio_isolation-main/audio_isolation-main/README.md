# Audio Isolation #
### Aim : ### 
Filtering individual voice components from audio inputs.
### Abstract : ###
The human auditory system is remarkable in its ability to function in busy acoustic environments. It is able to selectively focus attention on and extract a single source of interest in the midst of competing acoustic sources, reverberation and motion. Yet this problem, which is so elementary for most human listeners has proven to be a very difficult one to solve computationally. Even more difficult has been the search for practical solutions to problems to which digital signal processing can be applied. Many applications that would benefit from a solution such as hearing aid systems, industrial noise control, or audio surveillance require that any such solution be able to operate in real time and consume only a minimal amount of computational resources.
### What would the project be about? : ###
In the project we would try to build and train some machine learning models to filter out various components of sound from an auditory input. Our main aim would be to kind of solve the cocktail party problem and find and explore different ways in the process. Currently we aim at finding and implementing 3 different solutions to the problem. These approaches would be the following :
1. Independent Component Analysis(ICA)
2. CNN autoencoders model
3. Deep Mask Estimation
## About the Cocktail Party Problem ##
Cocktail Party Problem or the Blind Source Separation is a classical problem. The motivation for this problem is, imagine yourself in a party with a lot of people. There will be a sort of cacaphony becasue of all the people taling at the same time. Now you can shut out the voices in the background to hear some specific conversation. We also want to do the same and let's formalize what we are trying to do. Let's say there are m sources which are producing signal according to some distribution independent of each other and we have n microphones which record the signals arriving at them. We try to decipher the underlying source signals from the mixed signals arriving at the microphones. We will try and constraint the problem a little more so that we can move forward. The first assumption is that the mixed signals are a linear combination of the source signals. The second assumption is that the number of source and number of microphones are equal.
## Independent Component Analysis ##
*(model link: https://github.com/Monochrome901/audio_isolation/blob/main/ICA/ICA.ipynb)*


### Dataset: ###
The three audio files that make up the dataset each contain a mixed recording of three audio components  where each   recording is made from a different view.
Audio Files: 
1. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/audio11.wav
2. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/Audio22.wav
3. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/Audio33.wav

### About the Model: ###
Independent Component Analysis (ICA) is a technique that allows the separation of a mixture of signals into their different sources. In the basic formulation of ICA it is assumed that the received mixtures are instantaneous linear combinations of the source signals. ICA separation of mixed signals gives very good results is based on two assumptions and three effects of mixing source signals. Two assumptions:
1. The source signals are independent of each other.
2. The values in each source signal have non-Gaussian distributions.

Three effects of mixing source signals:
1. Independence: As per assumption 1, the source signals are independent; however, their signal mixtures are not. This is because the signal mixtures share the same source signals.
2. Normality: According to the Central Limit Theorem, the distribution of a sum of independent random variables with finite variance tends towards a Gaussian distribution.Loosely speaking, a sum of two independent random variables usually has a distribution that is closer to Gaussian than any of the two original variables. Here we consider the value of each signal as the random variable.
3. Complexity: The temporal complexity of any signal mixture is greater than that of its simplest constituent source signal.
Those principles contribute to the basic establishment of ICA. If the signals extracted from a set of mixtures are independent, and have non-Gaussian histograms or have low complexity, then they must be source signals.

Here, there are two people speaking simultaneously in a room and their voices (S1 and S2) are recorded by two microphones placed at different locations. We assume that the two voices (S1 and S2) are non-Gaussian and statistically independent. Their linearly mixed signals are recorded as X1 and X2 that can be expressed by the following linear equation: 

x = As

Both, A and s are unknown and we try to estimate them by ICA. The parameters (a11, a12, a21, a22) in matrix A are related to the distances between the microphones and the two speakers. ICA is aimed to estimate A and s, and obtain a de-mixing matrix W. For simplicity, we assume that the unknown mixing matrix A is square. The goal is to recover the original people’s voices (S1 and S2) when we are only given the observed data (i.e., X1 and X2). After estimating the matrix A, we can compute its inverse W and obtain the independent components u1 and u2(estimated sources) as follows:

s = u = Wx

For the model we use Fast-ICA. Fast-ICA algorithm is most widely used method for blind source separation problems, it is computationally efficient and requires less memory over other blind source separation algorithm for example infomax. The other advantage is that independent components can be estimated one by one which again decreases the computational load.

The algorithm goes as follows:
1. Centering the data(i.e. subtracting mean of a row from each element of the row).
2. Whitening the data
3. Single component extraction,
To know more about the algorithm you might want to checkout this <a href="https://en.wikipedia.org/wiki/FastICA">wikipedia</a> article
### Results: ###
Audio Files:
1. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/result_wav_11.wav
2. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/result_wav_22.wav
3. https://github.com/Monochrome901/audio_isolation/blob/main/ICA/result_wav_33.wav
### References: ###
1. https://medium.com/@ssiddharth408/cocktail-party-problem-using-unsupervised-learning-97fc665a4e94
2. https://github.com/vishwajeet97/Cocktail-Party-Problem
3. https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
4. https://www.researchgate.net/publication/337830429_Fusion_of_multi-physics_data_with_machine_learning_independent_component_analysis_ICA_improves_detection_of_geological_features/download
5. https://arnauddelorme.com/ica_for_dummies/
6. https://en.wikipedia.org/wiki/Independent_component_analysis
7. https://en.wikipedia.org/wiki/FastICA
## CNN Autoencoder Model ##
*(model link: )*

### Dataset: ###


### About the Method: ### 
Only the amplitude values of the signal across time are available to us as we visualise the waveform or time-domain signal. We could extract features like envelopes, RMS values, zero-crossing rate, etc., but these features are too basic and insufficiently discriminative to aid in the problem's solution. In order to begin, we must somehow reveal the structure of human speech if we are to extract voice material from a mix. The Short-Time Fourier Transform (STFT) comes to our aid, fortunately.

We know we can represent an audio signal ‘as an image’ using the Short-Time Fourier Transform right? Even though these audio images don’t follow the statistical distribution of natural images, they still expose spatial patterns (in the time vs frequency space) that we should be able to learn from.

The encoding stage consists of three layers:
1. a vertical convolution layer to obtain the frequency feature, 
2. a horizontal convolution layer to obtain the time-dependent feature and produce a time-frequency encoding, and 
3. a fully connected layer with a Rectified Linear Unit (ReLU) as the activation that shares its output with the layers in the decoding stage. For minimal information loss and with a ReLU as its activation function, the layers in each of the four decoding stages per class correspond to 
    1. a fully connected layer that shares the same dimensions as the encoding horizontal convolution layer,
    2. a horizontal deconvolution layer, and 
    3. a vertical deconvolution layer that shares the opposite dimensions of the horizontal and vertical convolution layers in the encoding stage. A ReLU is then applied to the output of each of the five decoding stages before being utilised for loss estimation and back propagation.

The model tries to train on the amplitude of the input STFT while ignoring its phase and is given a mixed source input and five isolated source outputs, all of which have undergone feature extraction and normalisation. It was discovered that training only on magnitude and passing the input phase to each of the five outputs, as opposed to training on both the magnitude and the phase of the input STFT, resulted in a negligible difference in the signal quality between (a) output signals with their respective magnitude and phase and (b) output signals with their respective magnitude but input phase. Comparing this to training on both the magnitude and phase of the input STFT resulted in a factor of 2 reduction in the number of parameters to be trained over. After then, the model tries to learn a set of five distinct filters that are applied to the input to produce five outputs that represent the isolated sources,which are calculated as: 
* Extract each of the five classes from the concatenated model output, oi 
* Divide each of the extracted outputs by the total sum to obtain the filters
* Multiply each of the filters to the input to obtain the predicted isolated output
     
Once the filters are applied to the input to obtain the five calculated outputs, the losses are computed per class by measuring the Mean Squared Error (MSE) between the calculated and the provided 3 target output. The individual losses are then summed up together for the model to attempt to optimize using the ADAM optimization algorithm.
### Results: ###



### References: ###
1. https://towardsdatascience.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785
2. https://github.com/ahpvjk/audio-classification-and-isolation
3. https://towardsdatascience.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de
