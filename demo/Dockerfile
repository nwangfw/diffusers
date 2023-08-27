# Use an official Python runtime as the parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime


RUN apt-get update \
    && apt-get install -y git git-lfs vim wget curl python3-pip python3-venv



# Install required packages and libraries
RUN pip install --no-cache-dir --upgrade git+https://github.com/huggingface/diffusers.git && \
    pip install --no-cache-dir -U transformers accelerate scipy datasets bitsandbytes


# Create a directory for the script and set it as the working directory
WORKDIR /app

# Copy the training script into the container
# Note: You need to have 'train_text_to_image.py' in the same directory as this Dockerfile

#RUN git clone --branch main --single-branch https://huggingface.co/CompVis/stable-diffusion-v1-4
#RUN 
COPY train_text_to_image.py /app/
#COPY pokemon-blip-captions /app/
#COPY stable-diffusion-v1-4 /app/

# The command that will be executed when the container starts
# CMD ["accelerate", "launch", "--mixed_precision=fp16", "train_text_to_image.py", \
#      "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4", \
#      "--dataset_name=lambdalabs/pokemon-blip-captions", \
#      "--use_ema", \
#      "--resolution=512", "--center_crop", "--random_flip", \
#      "--train_batch_size=1", \
#      "--gradient_accumulation_steps=4", \
#      "--gradient_checkpointing", \
#      "--max_train_steps=15000", \
#      "--learning_rate=1e-05", \
#      "--max_grad_norm=1", \
#      "--lr_scheduler=constant", "--lr_warmup_steps=0", \
#      "--output_dir=sd-pokemon-model", \
#      "--use_8bit_adam"]

CMD ["accelerate", "launch", "--mixed_precision=fp16", "train_text_to_image.py", \
     "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4", \
     "--dataset_name=lambdalabs/pokemon-blip-captions", \
     "--use_ema", \
     "--resolution=512", "--center_crop", "--random_flip", \
     "--train_batch_size=1", \
     "--gradient_accumulation_steps=4", \
     "--gradient_checkpointing", \
     "--max_train_steps=15000", \
     "--learning_rate=1e-05", \
     "--max_grad_norm=1", \
     "--lr_scheduler=constant", "--lr_warmup_steps=0", \
     "--output_dir=sd-pokemon-model", \
     "--use_8bit_adam"]