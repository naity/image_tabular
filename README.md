# image_tabular
> Integrate image and tabular data for deep learning.


## Install

`pip install image_tabular`

## How to use

This library utilizes fastai and pytorch to integrate image and tabular data for deep learning and train a joint model using the integrated data.

![title](pics/model.png)
<center>Image source: N. Gessert, M. Nielsen and M. Shaikh et al. / MethodsX 7 (2020) 100864<center>

1. Please first prepare image and tabular data separately as fastai LabelLists, and then integrate them using the `get_imagetabdatasets` function as below:  

```python
integrate_train, integrate_valid, integrate_test = get_imagetabdatasets(image_data, tab_data)
```  

2. The train, validation, and optional test datasets can then be combined in a DataBunch:  
```python
db = DataBunch.create(integrate_train, integrate_valid, integrate_test,
                      path=data_path, bs=bs)
```

3. Next, we create a joint model to train on both image and tabular data simultaneously:
```python
integrate_model = CNNTabularModel(cnn_model,
                                  tabular_model,
                                  layers = [cnn_out_sz + tab_out_sz, 32],
                                  ps=0.2,
                                  out_sz=2).to(device)
```

4. Finally, we pack everying into a fastai learner and train the joint model:
```python
learn = Learner(db, integrate_model)
learn.fit_one_cycle(10, 1e-4)
```

The following notebook puts everything together and shows how to use the library for the SIIM-ISIC Melanoma Classification competition on Kaggle:  

[SIIM-ISIC Integrated Model](siim_isic_integrated_model.ipynb#siim_isic_integrated)
<a id='siim_isic_integrated'></a>
