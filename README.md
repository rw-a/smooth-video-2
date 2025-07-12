# Smooth Video 2: An Improved Video Frame Interpolator

## Introduction
 - Smooths video by interpolating frames (e.g. 24fps → 48fps). 
 - Useful for making cinematic slow-mo footage.
 - Based on a near state-of-the-art frame interpolator.
 - This is a more user-friendly adaptation of [this](https://github.com/MCG-NJU/VFIMamba) implementation.

<p float="left">
  <img src=figs/out_2x.gif width=340 />
  <img src=figs/out_8x.gif width=340 /> 
</p>

## Requirements
 - Linux (WSL works too)
 - Docker
 - CUDA GPU
 - git-lfs

## Installation
1. Clone the repository
```bash
git clone https://github.com/rw-a/smooth-video-2.git
cd smooth-video-2
```
2. Build and deploy the provided Docker image
```bash
docker compose up --build
```
3. Copy your video file into `mount_data`
```bash
# Assumes your file is called 'input.mp4'
cp input.mp4 mount_data
```
4. Run a shell inside the Docker container, then run the interpolator
```bash
docker compose exec smooth-video bash
python interpolate_video.py --model VFIMamba_S --n 2 --scale 0.5 mount_data/input.mp4 mount_data/output.mp4
```

> The model can be either `VFIMamba` or `VFIMamba_S`. `VFIMamba_S` is much faster but less accurate.

> `n`=2 means we double the framerate of the video. `n`=4 means we have 4 times the number of frames in total.

> The `scale` parameter improves inference speed. We downsample the images by the scale to predict the optical flow, then resize to the original size to perform the other operations.
> We recommend setting the scale to 0.5 for 2K frames and 0.25 for 4K frames.

## Credit

This fork is a more user-friendly adaptation of the [VFIMamba](https://github.com/MCG-NJU/VFIMamba) implementation, which is based on the following paper:

```
@misc{zhang2024vfimambavideoframeinterpolation,
      title={VFIMamba: Video Frame Interpolation with State Space Models}, 
      author={Guozhen Zhang and Chunxu Liu and Yutao Cui and Xiaotong Zhao and Kai Ma and Limin Wang},
      year={2024},
      eprint={2407.02315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.02315}, 
}
```

Please cite this paper if you find it useful in your research or application.

## License and Acknowledgement

This project is released under the Apache 2.0 license. The original implementation is [VFIMamba](https://github.com/MCG-NJU/VFIMamba), which is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [EMA-VFI](https://github.com/whai362/PVT), [MambaIR](https://github.com/csguoh/MambaIR?tab=readme-ov-file#installation) and [SGM-VFI](https://github.com/MCG-NJU/SGM-VFI). Please also follow their licenses. Thanks for their awesome work.
