
# RoCourseNet: Distributionally Robust Training of a Prediction Aware Recourse Model

This repository implements paper "RoCourseNet: Distributionally Robust Training of a Prediction Aware Recourse Model" (submitted to Neurips 2022). 

## Requirements

1. install [pytorch](https://pytorch.org/)
2. install all dependencies

```
pip install -e
```

## Training
```
# train RoCourseNet
python -m scripts.train_cfnet --n_steps 13 
# train CounterNet
python -m scripts.train_cfnet --model_name CounterNet
# train RoCourseNet on the German Credit dataset
python -m scripts.train_cfnet --data_name german_credit

```


## Evaluation

### Evaluate CF Baselines

```
python -m scripts.eval_cf_baselines
```

### Evaluate VDS Attacker
```
python -m scripts.eval_attacker --data_name "loan" --model_name AdvCounterNet
```

## Pretrained Models

See `assets/weights`


## Useful `nbdev` commands

### Build nbs to module

```
nbdev_build_lib
```

### Update nbs from module
```
nbdev_update_lib
```

### clean notebooks
```
nbdev_clean_nbs
```
