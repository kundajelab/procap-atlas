python attribute_bpnet_ccre.py -e ENCSR359PWL -v --ohe -b 16
python attribute_bpnet_ccre.py -e ENCSR359PWL -v --ohe --head count -b 16
python attribute_bpnet_gc.py -e ENCSR359PWL -v -b 16
python attribute_bpnet_gc.py -e ENCSR359PWL -v --head count -b 16

for head in count profile; do echo python attribute_bpnet.py -e ENCSR220XSM -b ccre -v --head ${head}; done | simple_gpu_scheduler --gpus 1 3
for head in count profile; do echo python attribute_bpnet.py -e ENCSR220XSM -b ccre_gc -v --head ${head} -bs 16; done | simple_gpu_scheduler --gpus 2 3
for head in count profile; do echo python attribute_bpnet.py -e ENCSR220XSM -b dnase -v --head ${head} -bs 16; done | simple_gpu_scheduler --gpus 2 3
for head in count profile; do echo python attribute_bpnet.py -e ENCSR220XSM -b gc -v --head ${head} -bs 16; done | simple_gpu_scheduler --gpus 0 1

