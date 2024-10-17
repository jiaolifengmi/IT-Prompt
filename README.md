#  VQ-Prompt
Official PyTorch code for "Prompt-based Continual Learning for Extending Pretrained CLIP Models' Knowledge (ACMMM Asia2024)".


<p align="center">
<img src="method_it-p.png" width="90%">
</p>

## Abstract
Contrastive Language-Image Pretraining (CLIP) model has demonstrated remarkable performance and strong zero-shot capabilities through its training on text-image datasets using contrastive learning. This has sparked interest in developing continuous learning methods based on CLIP to extend its knowledge to new datasets. However, traditional continuous learning approaches often involve modification to the original parameters of the pretrained CLIP model and consequently compromise its zero-shot capabilities. Additionally, the substantial parameter size of the CLIP model makes it challenging for traditional continuous learning methods due to lengthy training times. To tackle these challenges, we propose Image Text (IT-)Prompt, which leverages the inherent correlation between visual and textual information to train discrete prompts dedicated to individual tasks, serving as repositories for task-specific knowledge. By employing discrete textual prompts as guidance, we ensure the uniqueness of each task's prompt and prevent interference among tasks, thus alleviating catastrophic forgetting during continuous learning. While retaining the pretrained parameters of CLIP, our approach introduces only a small number of additional trainable parameters. This allows us to enhance training efficiency and preserving the original zero-shot capabilities of CLIP. Comparative experiments show that IT-Prompt achieves a performance improvement of at least 10\% compared to state-of-the-art methods..


## Requirements
 * python=3.8.18
 * torch=2.0.0+cu118
 * torchvision=0.15.1+cu118
 * timm=0.9.12
 * scikit-learn=1.3.2
 * numpy
 * pyaml
 * pillow
 * opencv-python
 * pandas
 * openpyxl (write results to a xlsx file)

 
## Datasets
 * Create a folder `datasets/`
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

## Checkpoints
 * Create a folder `pretrained/`
 - [iBOT-21k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint.pth)
 - [iBOT-1k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
 - [DINO-1k](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)
 - [MoCo-1k](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar)

## Training
**The complete code will be uploaded as soon as possible**.

## Results
Results will be saved in a folder named `output/`.

## Reference Codes
[1] [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)

## Citation
If you find this repository is useful, please cite the following reference.
```
@article{jiao2024,
  title={Prompt-based Continual Learning for Extending Pretrained CLIP Models’ Knowledge},
  author={Jiao, Li and Cao, Lihong and Wang, Tian},
  journal={ACMMM Asia},
  year={2024}
}
```
