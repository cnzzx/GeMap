# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Eval

The general command for training and evaluation is
```
bash tools/[script name] projects/configs/gemap/[config name] [checkpoint] [the number of GPUs] [port]
```
where "checkpoint" is used only for evaluation. Before evaluation, you should train a model or download our pretrained checkpoints (listed in [README](../README.md)) first . We provide some examples in the following.

Train GeMap with 8 GPUs 
```
bash tools/dist_train.sh projects/configs/gemap/gemap_simple_r50_110ep.py 8 20000
```

Evaluate GeMap with 1 GPU
```
bash tools/dist_test_map.sh projects/configs/gemap/gemap_simple_r50_110ep.py ckpts/gemap_simple_r50_110ep.pth 1 20000
```

Evaluate GeMap under different weather conditions with 1 GPU
```
bash tools/dist_test_map_weather.sh projects/configs/gemap/gemap_simple_r50_110ep.py ckpts/gemap_simple_r50_110ep.pth 1 20000
```

\* The weather evaluation results will be saved to ''work_dirs/[config name]/weather" by default.

# Test FPS
To test the FPS, you can reference the following command (note that "checkpoint" is not necessary). 
```
python tools/analysis_tools/benchmark.py projects/configs/gemap/gemap_simple_r50_110ep.py \
--checkpoint ckpts/gemap_simple_r50_110ep.pth
```
As the hardware and software conditions are very complex, the FPS might vary a lot even with the same type of GPU. Results reported in [README](../README.md) are only for reference.