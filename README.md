# Multimodal Architecture for Video Captioning with Memory Networks and Attention Mechanism
Tensorflow implementation of the framework described in the following paper : [Multimodal Architecture for Video Captioning with Memory Networks and Attention Mechanism
](https://www.sciencedirect.com/science/article/pii/S016786551730380X)

## Prerequisites

* Python 3.3+
* Tensorflow 1.1.0
* NumPy
* pandas

## Generating Data

The MSVD dataset can be download from [here](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

Update required fields and run the following commands:

```
python3 generate_nolabel.py
```

```
python3 input_generator.py
```

```
python3 trans_video_youtube.py
```

## Usage

```
python3 Main_h.py
```

## Disclaimer

The codes from the following repositories were referred:

* [carpedm20/NTM-tensorflow](https://github.com/carpedm20/NTM-tensorflow)
* [tsenghungchen/SA-tensorflow](https://github.com/tsenghungchen/SA-tensorflow)
