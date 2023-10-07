---
layout: post
title: "MRI Super-Resolution with Ensemble Learning and Complementary Priors"
categories: [Medical Imaging]
year: 2015
type: paper
author: Lyu
exturl: https://arxiv.org/pdf/1907.03063.pdf
---
MRI suffers from major physical limitations in background magnetic field, gradient fields and imaging speed. As a result of this, the spatiotemporal quality of MR images is lackluster and the only way to obtain higher resolution images is to use stronger background magnetic fields or increase the scan time, both of which are not clinically applicable solutions. Super-resolution is a post-processing technique that can be applied outside the clinical environment to enhance the resolution of MR images. This paper approaches the problem of up-scaling 320x320 images from the NYU fastMRI dataset.

### Architecture

![](/images/mriensemble.png)

During training, HR images are down-sampled in the fourier domain becoming LR images. Before the LRIs are fed into the neural networks there is a processing step which enlarges the images into *processed LR images* (PLR). 5 classic model-based algorithms are used to enlarge the LRIs and obtain corresponding PLRs. Next, the PLRs are fed along to a GAN (all with the same architecture) which generates a prediction result and then uses a discriminator to differentiate between the prediction and the original HR. Out from the GAN SR process comes the 5 different SR predictions. Finally, a CNN model is used to integrate all these images through ensemble learning. 

### Results / Takeaways

Structural similarity (SSIM) and peak signal-to-noise ratio (PNSR) are used to evaluate image quality. By empirical analysis, the PLR images had smoother edges and clearer shapes when compared to the LR images. Although some artifacts are introduced it's clear that the 5 algorithms each have their own independent strengths and weaknesses. The 5 GAN models, each separately trained, manages to further improve the PLR images but some artifacts are still persistent at this stage. Looking at the metrics in Table III we notice that it records only a minor improvement from the PLR images. What stood out most to me was the success of the final step in the SR process, ensemble learning. On average it manages to improve SSIM by ~10% compared to the GAN SR predictions. Each of the individual GAN SR predictions contain substantial artifacts, but because they were derived from processed images that were obtained from different approaches they were unlikely to be the same or coherent. Through ensemble, these artifacts were removed or at least greatly reduced. I really like the idea of taking an original image and deriving 5 datasets from it all with their own slight differences and then combining them through a CNN-based integrator which learns to perform ensemble learning in a data-driven fashion. 

