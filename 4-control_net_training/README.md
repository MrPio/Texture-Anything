# Train the ControlNet

Now that we have generated our dataset as a set of triplets (`UV`, `Caption`, `Diffuse`), we need to train a ControlNet to control the diffusion process with Canny UV maps.

> Basically, the network needs to learn the function $f(\text{Caption},\text{UV})=\text{Diffuse}$

## Devising a training strategy

In order to establish a successful training, we first need to address the following questions:

1. Which training script should we use?
1. Should we start from scratch or use a pretrained Canny ControlNet?
1. What type of diffusion model should we use? Stable Diffusion or Flux? If Stable Diffusion, then which version? `1.5`, `XL`, `2.1` or `3.0`?

### The training script
Following [Train your ControlNet with diffusers 🧨](https://huggingface.co/blog/train-your-controlnet), we choose to employ [the scripts](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) provided by the diffusers repository.

### The ControlNet model
Even though the domain is different, we decide to load a pretrained Canny ControlNet instead of training from scratch.

### The choice of the diffuser model
Since `SD 1.5` is quite basic and `Flux` is [way too memory expensive](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_flux.md) to employ, we choose `SD 2.0`.