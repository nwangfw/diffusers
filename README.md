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

```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
```
Then
```
accelerate launch  train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME  --dataset_name=$DATASET_NAME  --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing     --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0  
```


Notes:
You can modify some parameters based on your machine's capability'


