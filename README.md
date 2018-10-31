## How SGD Selects the Global Minima in Over-parameterized Learning: A Dynamical Stability Perspective
by Lei Wu, Chao Ma, Weinan E


### Training
```
python train.py --dataset fashionmnist --training_size 1000 --model_file net.pkl
```

### Computing sharpness and non-uniformity
```
python diagnose.py --dataset fashionmnist --training_size 1000 --model_file net.pkl
```


### Dependencies
- pytorch >= 0.4

### Citation

	@inproceedings{leiwu2018,
		title={How SGD Selects Global Minima in Over-parameterized Learning: A Dynamical Stability Perspetive},
		author={Wu, Lei and Ma, Chao and E, Weinan},
		booktitle={Advances in Neural Information Processing Systems},
		year={2018}
	}
