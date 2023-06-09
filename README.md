# 16726_p5

## 1. Generator inversion

We compare
- StyleGAN (z, w, w+) vs Vanilla GAN (z)
- From our experiments with StyleGAN in HW4, we judge the default of conv_5 perceptual loss to be reasonable, so we directly compare combinations of Lp (L1), perceptual loss, and L2 of delta regularization. We also note initializing from an average of many samples hurts performance, especially for Vanilla GAN, which barely starts learning, so we keep this option off.

We compare latent/model combo at a time, across hyperparameters (2^3 choices of losses to include). We first note early and final results. The wandb report is [here](https://wandb.ai/joelye9/16726_p5/reports/Part-1-Inverse-Projection--Vmlldzo0MDc1NjE3). The loss hyperparameters are
```
- Lp/L1 (default weight 10)
- Perceptual (default weight 0.1)
- L2 norm (tuned to 1e-3 in early exps)
```

### Target Image
![p1_target](./figures/p1_data.png)

### Vanilla: Z
![p1_vanilla_z](./figures/p1_vanilla_z.png)

### StyleGAN: Z

![p1_stylegan_z](./figures/p1_stylegan_z.png)

### StyleGAN: W

![p1_stylegan_w](./figures/p1_stylegan_w.png)

### StyleGAN: W+

![p1_stylegan_w+](./figures/p1_stylegan_wp.png)


### Commentary
Universally, results are not good without perceptual loss -- the second row is always less faithful than the first. Including L1 loss on top of this improves convergence rates - the first two images of each group are pretty close to the rigth result. L2 norm has the expected effect of subduing some of the green background, though cat likeness is not obviously affected.

Vanilla GAN converges very rapidly, interestingly, but the quality is not good.


## 2. Interpolation
We trim hyperparameters to now include both losses all the time, but do still toggle regularization. We show no regularization, and then with regularization try the four models.

### Vanilla: Z

![p2_vanilla_z_noreg_1](./output/interpolate/1_vanilla_z_0.gif)
![p2_vanilla_z_reg_1](./output/interpolate/1_vanilla_z_0.001.gif)

![p2_vanilla_z_reg_3](./output/interpolate/3_vanilla_z_0.gif)
![p2_vanilla_z_reg_3](./output/interpolate/3_vanilla_z_0.001.gif)

The interpolation does still appear to work in so far as the in-between samples are still reasonable cat samples for the Vanilla GAN.  Minimal perspective shifting ibilities.

### StyleGAN: Z

![p2_stylegan_z_noreg](./output/interpolate/1_stylegan_z_0.gif)
![p2_stylegan_z_reg](./output/interpolate/1_stylegan_z_0.001.gif)

![p2_stylegan_z_noreg](./output/interpolate/3_stylegan_z_0.gif)
![p2_stylegan_z_reg](./output/interpolate/3_stylegan_z_0.001.gif)

The interpolation transitions unsatisfactorily for the first pair, particularly when the blue eyes must warp out of existence. Regularization doesn't seem to have a major effect on this.

### StyleGAN: W

![p2_stylegan_w_noreg](./output/interpolate/1_stylegan_w_0.gif)
![p2_stylegan_w_reg](./output/interpolate/1_stylegan_w_0.001.gif)

![p2_stylegan_w_noreg](./output/interpolate/3_stylegan_w_0.gif)
![p2_stylegan_w_reg](./output/interpolate/3_stylegan_w_0.001.gif)

The first pair still has a pretty unsatisfactory interpolation.  Really, the onset of the second image is sudden, it doesn't appear like a continuous change.

### StyleGAN: W+

![p2_stylegan_w+_noreg](./output/interpolate/1_stylegan_w+_0.gif)
![p2_stylegan_w+_reg](./output/interpolate/1_stylegan_w+_0.001.gif)

![p2_stylegan_w+_noreg](./output/interpolate/3_stylegan_w+_0.gif)
![p2_stylegan_w+_reg](./output/interpolate/3_stylegan_w+_0.001.gif)

Better than the rest. The onset of the second image appear to lapse over more samples, which is something.

## 3. Scribbles

I drew my own sample but for some reason the dataloader didn't respect it :( and just rolled with the given sketches. Here we just use StyleGAN W+ which produced the best results in the previous section.

### Sketch 1

![p3_sketch_1](./output/draw/0_data.png)
![p3_synth_1](./output/draw/0_stylegan_w+_perc-0.01_l1-10-l2-0.001_8000.png)

### Sketch 2 (my own sample, but bugged?)

![p3_sketch_2](./output/draw/1_data.png)
![p3_synth_2](./output/draw/1_stylegan_w+_perc-0.01_l1-10-l2-0.001_8000.png)

Yeah, this looks like the first sketch first some reason... (prob a dataloader bug)

### Sketch 3

![p3_sketch_3](./output/draw/2_data.png)
![p3_synth_3](./output/draw/2_stylegan_w+_perc-0.01_l1-10-l2-0.001_8000.png)

### Sketch 4

![p3_sketch_4](./output/draw/3_data.png)
![p3_synth_4](./output/draw/3_stylegan_w+_perc-0.01_l1-10-l2-0.001_8000.png)

### Sketch 5
![p3_sketch_5](./output/draw/4_data.png)
![p3_synth_5](./output/draw/4_stylegan_w+_perc-0.01_l1-10-l2-0.001_8000.png)

It's apparent from the poor quality of sketch 4 and 5 that the optimization does heavily weight the constraints. Hence, lighter guides are visually more pleasing. In some cases it looks like cat features are pasted on wherever the guide isn't. It's a bit confusing to me how these outputs are even on the manifold at all.

However this intuition has holes. For example in image 1, only some whiskers are drawn, but the generated whiskers are many more and moreover don't even appear to be in the same place as the guidance. Whiskers are also possibly trying to appear even in the dense sketches. It is possible the guidance is simply ignored in this case, and manifold adherance reigns.