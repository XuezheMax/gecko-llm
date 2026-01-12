<div align="center">
   <img src="./assets/logo.png" width="200"><br><br>
</div>

-----------------------------------------------

# Gecko
Reference implementation of Gecko 7B model.

>[Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths]()

>Xuezhe Ma*, Shicheng Wen*, Linghao Jin*, Bilge Acun*, Ruihang Lai*, Bohan Hou, Will Lin, Hao Zhang, Songlin Yang, Ryan Lee, Mengxi Wu, Jonathan May, Luke Zettlemoyer, Carole-Jean Wu

Discord: []()

## Updates
1. [Jan 12th 2026] Release Repo to public.

## Installation
First install PyTorch >= 2.8.0 with cuda 12.8
```bash
pip install torch torchvision
```

Then, install gecko-llm
```bash
https://github.com/XuezheMax/gecko-llm.git
cd gecko-llm
pip install -r requirements.txt
pip install -e .
```

## References
```

@article{ma2024megalodon,
  title={Megalodon: Efficient llm pretraining and inference with unlimited context length},
  author={Ma, Xuezhe and Yang, Xiaomeng and Xiong, Wenhan and Chen, Beidi and Yu, Lili and Zhang, Hao and May, Jonathan and Zettlemoyer, Luke and Levy, Omer and Zhou, Chunting},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={71831--71854},
  year={2024}
}

@inproceedings{
  ma2023mega,
  title={Mega: Moving Average Equipped Gated Attention},
  author={Xuezhe Ma and Chunting Zhou and Xiang Kong and Junxian He and Liangke Gui and Graham Neubig and Jonathan May and Luke Zettlemoyer},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
}
```
