# megatron-lm-recipes
Achieving the SOL training performance with Megatron-LM.

Now we provide scripts for training llama2 series models.
## Usage
* Pull the latest commit on the main branch of Megatron-LM repo
  ```
  git clone https://github.com/NVIDIA/Megatron-LM.git
  ```
* Pull the latest commit on the main branch of TransformerEngine repo and install
  ```
  git clone https://github.com/NVIDIA/TransformerEngine.git
  ```
* Create a config file including your home path and wanbd api key
  ```
  !/bin/bash
  HOME_PATH=***
  WANDB=***
  ```
* Modify the relative path in the scripts
* Prepare the dataset following the instructions [here](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron?tab=readme-ov-file#data-preprocessing) or use mock data
* Run training inside NeMo Framework container v24.01.
  ```
  nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.01
  ```

## Note
* The scripts are designed for slurm environment, and coordinated with the related slurm scripts. Hence, for other environments like K8s, you need to modify the launch cmd for running on multiple nodes.
* We verified the performance on the following commits
  * Megatron-LM: ```361c97d4bb8f749fd55035705a811616ec085ed2 (2/20/2024)```
  * TransformerEngine: ```2187a8f3bbadff7fb9922b85473a6135f8272fdc(2/20/2024)```
