# Towards A Masked Loss
The loss function should not account for zones of the texture that are not mapped by the UV map.

## 📌 TL;DR
I needed a way to mask the MSE loss computed in the latent space. I thought it was impossible to do without exiting the latent space because feeding the boolean mask into the VAE would twist it into a meaningless float tensor.
_That couldn't be further from the truth!_ The VAE uses a convolutional encoder that **preserves the spatiality of the information**. You can safely downsample the mask with nearest interpolation and use it in the latent space.

## What is a VAE?
Like GANs, _autoencoders_ are a kind of generative networks.

### AE vs VAE
* A `simple autoencoder` is trained to _encode_ and _decode_ with the minimum information loss, regardless any kind of constraint of how the latent space is structured. Therefore, a form of regularization is needed to ensure that sampling and decoding a point from the latent space yields meaningful data.

* A `varational autoencoder` **avoids overfitting through regularization**. To do this, **the encoder outputs a PDF** $q(z\mid x)$ rather than a point $z\in \mathbb{R}^L$. Then, during training, a $z$ is sampled from $q$ and then passed through the decoder. This makes the latent space "smoother," so we can expect **similar images to be decoded from nearby latent points** (semantics correspondence).

>[!NOTE]
>**The stochastic behaviour of VAE**
>
>The VAE preserves the deterministic autoencoder structure made of a convolutional bottleneck. The stochastic behaviour is introduced through sampling, which occurs solely during training on the PDF predicted by the encoder.

### The Kullback-Leibler divergence
The chosen PDF is _Gaussian_, and the loss is given by the sum of the mean squared error (MSE) between the decoded latent sample and the input and the Kullback-Leibler divergence.

$$ \mathcal L_{VAE}= \mathcal L_{\text{reconstruction}} + \mathcal L_{\text{regularizer}} $$

This divergence pulls the predicted Gaussian PDF (the predicted mean and variance) to assume the form of a standard Gaussian, which we assume to be the true probability distribution. This prevents the formation of small island in the latent domain.

### The architecture of a VAE
Stable Diffusion's VAE uses 3 convolution layers with stride 2 (and opportune padding), meaning that the output latent is a $8\times$ downsampled version of the pixel-space image ($64\times$ information reduction). The number of channels is instead increased to $C_{feat}=16$ or more.

The output of the 3 down blocks, of shape $[B,\,C_{feat},\,64,\,64]$, is passed through 2 fully connected layers (four $C_{feat}\times 1\times 1$ kernels that reshape the tensor $[B,\,C_{feat},\,64,\,64]$ to two $[B,\,4,\,64,\,64]$ tensors, one for the mean and the other for the variance) to predict, for each "latent pixel", a $\mu$ and a $log(\sigma)$ (for numerical stability). When used for inference, the second head is deactivated and **the latent point** $z\sim q(z\mid x)$ **is deterministically chosen as** $z=\mu$.

>[!NOTE]
>**Evolution of tensor shape in the VAE**
>
>* Input block: `[B, 3, 512, 512]` $\rightarrow$ `[B, 4, 512, 512]`.
>* 3 Conv layers: `[B, 4, 512, 512]` $\rightarrow$ `[B, >16, 64, 64]`.
>* Input block: `[B, >16, 64, 64]` $\rightarrow$ $2\times$ `[B, 4, 64, 64]`.

### The VAE is deterministic for inference.
Therefore, one could argue that the stochastic behavior of the VAE is eliminated when it is used for inference. This makes sense because sampling is used for training purposes to learn a robust latent representation that satisfies both the _continuity_ and _completeness_ properties.

## Conclusion
Now that we know how a VAE works, it's clear that we can simply downsample the pixel-space mask and multiply it with the noise prediction error.