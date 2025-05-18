# Train the ControlNet

Now that we have generated our dataset as a set of triplets (`UV`, `Caption`, `Diffuse`), we need to train a ControlNet to control the diffusion process with Canny UV maps.

> Basically, the network needs to learn the function $f(\text{Caption},\text{UV})=\text{Diffuse}$

## Devising a training strategy

In order to establish a successful training, we first need to address the following questions:

1. Should we start from scratch or use a pretrained Canny ControlNet?
2. What type of diffusion model should we use? Stable Diffusion or Flux? If Stable Diffusion, then which version? `1.5`, `XL`, `2.0` or `3.0`?
3. Which training script should we use?