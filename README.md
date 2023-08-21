Installation

```
git clone https://github.com/nwangfw/diffusers
cd diffusers
pip install .
```

```
cd ./examples/text_to_image
pip install -r requirements.txt
```

You need to have a huggingface account and get your token
```
huggingface-cli login
```
My huggingface token is 
```
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
```

launch command

Add environment values for each new terminal window
```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export 
```
Then
```
accelerate launch --mixed_precision="fp16" train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME  --dataset_name=$DATASET_NAME  --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=4   --gradient_accumulation_steps=4   --gradient_checkpointing    --use_8bit_adam  --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1  --lr_scheduler="constant"   --lr_warmup_steps=0 
```


Notes:
You can modify some parameters based on your machine's capability'


Command to get log 
```
tensorboard --logdir 
```


Misc:

Got the following error for large batchsize
```
RuntimeError: Expected is_sm80 to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
```