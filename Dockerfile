FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
RUN pip install transformers==4.43.0 tokenizers==0.19.1 trl==0.9.6

