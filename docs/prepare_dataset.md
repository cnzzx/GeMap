

## NuScenes
Download nuScenes V1.0 full dataset data  and CAN bus expansion data from offcial release [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/create_data.py nuscenes --root-path ./data/hdmap/nuscenes --out-dir ./data/hdmap/nuscenes/simple --extra-tag nuscenes --version v1.0 --canbus ./data/hdmap
# for the simple objective configuration

python tools/gemap/custom_nusc_map_converter.py --root-path ./data/hdmap/nuscenes --out-dir ./data/hdmap/nuscenes/full --extra-tag nuscenes --version v1.0 --canbus ./data/hdmap
# for the full objective configuration
```

Using the above code will generate `nuscenes_map_infos_temporal_{train,val}.pkl`, which contain local vectorized map annotations.

**Folder structure**
```
GeMap
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/hdmap/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── simple/
|   |   |   ├── nuscenes_infos_temporal_train.pkl
|   |   |   ├── nuscenes_infos_temporal_val.pkl
|   |   ├── full/
|   |   |   ├── nuscenes_infos_temporal_train.pkl
|   |   |   ├── nuscenes_infos_temporal_val.pkl
```

## Argoverse 2
Download the Argoverse 2 Sensor Dataset [here](https://www.argoverse.org/av2.html#download-link).

**Folder structure**
```
GeMap
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/hdmap/
│   ├── can_bus/
│   ├── nuscenes/
│   ├── av2/
│   │   ├── sensor/
|   |   |   |—— train/
|   |   |   |—— val/
|   |   |   |—— test/
|   |   |   ├── full/
|   |   |   |   ├── av2_map_infos_train.pkl
|   |   |   |   ├── av2_map_infos_val.pkl
```

**Prepare Argoverse 2 data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/gemap/custom_av2_map_converter.py --data-root ./data/hdmap/av2/sensor/ --out-dir ./data/hdmap/av2/sensor/full
# the simple and full objective configurations on AV2 share the same dataset pipeline
```

Using the above code will generate `av2_map_infos_{train,val}.pkl`, which contain local vectorized map annotations.
