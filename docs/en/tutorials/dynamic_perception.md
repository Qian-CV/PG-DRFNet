# Tutorial: How to Use Dynamic Perception

We incorporate the dynamic perception for inference, which ensure model accuracy while maximizing model efficiency. Our PG-DRFNet with dynamic perception can adjust its network structure or parameters according to the input in the inference stage.

## Positional guidance relationship for train

When you train a dynamic network, you should firstly modify the configs, e.g. `model.bbox_head.guidance_head_size`, `model.bbox_head.guidance_layer_train`, and `model.bbox_head.fp_object_scale`. You can establish the positional guidance relationship in train by executing the following command

```bash
python tools/train.py \
  --config path/to/your/configs/ \
  --work-dir work_dirs/PG-DRFNet \
  --cfg-options model.bbox_head.guidance_head_size=[C, C, 4, 1] model.bbox_head.guidance_layer_train=[1, 2] model.bbox_head.fp_object_scale=[[0, 16], [0, 32]]
```

you should modify C according to your channel number of the candidate feature

## Dynamic perception for inference

When you test your dynamic network, you should firstly modify the configs, e.g. `model.bbox_head.dynamic_perception_infer`, `model.bbox_head.dp_version`,`model.bbox_head.dp_threshold`, `model.bbox_head.context`, `model.bbox_head.layers_no_dp`, `model.bbox_head.layers_key_test`, and `model.bbox_head.layers_value_test`. You can perform the dynamic perception in inference by by executing the following command

```bash
python tools/train.py \
  --config path/to/your/configs/ \
  --work-dir work_dirs/PG-DRFNet \
  --cfg-options model.bbox_head.dynamic_perception_infer=True model.bbox_head.dp_version='v2' model.bbox_head.dp_threshold=0.1 model.bbox_head.context=4 model.bbox_head.layers_no_dp=[1, 2, 3] model.bbox_head.layers_key_test=[1, 2] model.bbox_head.layers_value_test=[0, 1]
```

The `dp_version` can be choose 'v1' or 'v2' version.
