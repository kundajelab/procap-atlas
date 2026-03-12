ENCSR789VGQ


for fold in {0..6}; do echo python fit_procapnet_gc.py -e ENCSR220XSM -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 3 # K562 PRO-cap
for fold in {0..6}; do echo python fit_bpnet_gc.py -e ENCSR220XSM -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 PRO-cap
for fold in {0..6}; do echo python fit_bpnet_ccre.py -e ENCSR220XSM -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 PRO-cap
for fold in {0..6}; do echo python fit_bpnet_dnase.py -e ENCSR220XSM -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 PRO-cap
for fold in {0..6}; do echo python fit_bpnet_ccre_gc.py -e ENCSR220XSM -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 PRO-cap

for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR359PWL -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 PRO-cap



for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR261KBX -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # K562 CoPRO
for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR046BCI -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # A673
for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR100LIJ -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # CACO2

for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR799DGV -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # MCF10A
for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR935RNW -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # CALU3
for fold in {0..6}; do echo python fit_bpnet.py -e ENCSR098LLB -f $fold -v; done | simple_gpu_scheduler --gpus 0 1 2 3 # HUVEC