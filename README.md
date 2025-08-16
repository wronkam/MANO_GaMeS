# MANO + GaMeS
Based on Waczyńska and Borycki's ["GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting"](https://arxiv.org/abs/2402.01459).

Done:
- mergere of movable MANO mesh with GaMeS
- flexibly config defined multi-net joint temporal parameter update encoder-decoder
- training from dynamic scenes
- multithread camera preloading enebling training on large datasets without whole dataset cashing or image loading lag
- visualization tools for camera alignment
- InterHands dataset camera realignment 

ToDo:
- addition (and potentially creation) of zero-shot hand pose estimator to remove reliance on noisy camera information
- temporal model update blending
- dynamic mesh densification
- re-enable splat removal, disabled in GaMeS
