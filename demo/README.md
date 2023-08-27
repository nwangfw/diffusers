This is used for training demo

Launch command for running the demo in termnial environment
```
accelerate launch --mixed_precision="fp16"  train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME   --dataset_name=$DATASET_NAME   --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd-pokemon-model" --use_8bit_adam
```

Launch command in docker
```
docker run --gpus=all --ipc=host  my-training-image:v3 
```


Lanuch command for finetune.py
```
accelerate launch --mixed_precision="fp16"  finetune.py   --pretrained_model_name_or_path=$MODEL_NAME   --dataset_name=$DATASET_NAME     --resolution=512 --center_crop    --train_batch_size=1   --gradient_accumulation_steps=4     --max_train_steps=15000   --learning_rate=1e-05    --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd-pokemon-model" 
```

Note that there is mirror code modification for finetune.py since the huggingface libraries has been updated