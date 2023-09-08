# stereo-reconstruction
This is an ongoing C++ implementation of stereo matching methods.

The program takes a pair of calibrated stereo images as input and returns their disparity map. We provide two different methods: naive matching [1] and PatchMatch Stereo [2].

## Requriments
- OPENCV > 3.2

## References
- [ 1 ] Bleyer, Michael, and Christian Breiteneder. "Stereo matching—State-of-the-art and research challenges." Advanced topics in computer vision (2013): 143-179.
- [ 2 ] Bleyer, Michael, Christoph Rhemann, and Carsten Rother. "Patchmatch stereo-stereo matching with slanted support windows." Bmvc. Vol. 11. 2011.
