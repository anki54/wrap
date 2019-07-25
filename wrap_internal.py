## Server only copy for wrap client.

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import *
from sklearn.neural_network._stochastic_optimizers import *
from sklearn.neural_network import *

def get_logistic_regression():
    return LogisticRegression()

def get_mlp_classifier():
    return MLPClassifier()

def getLabelBinarizer():
    return LabelBinarizer()

def getAdamOptimizer():
    return AdamOptimizer()

nparrays=['coef_','n_iter_','classes_']

recognized_models={
    'LogisticRegression':get_logistic_regression,
    'MLPClassifier':get_mlp_classifier
}

recognized_model_attr_map={
    'LabelBinarizer':getLabelBinarizer,
    'AdamOptimizer':getAdamOptimizer
}

std_type={'int','str','float','tuple','bool','float64','ndarray','list'}

def getClasAtr(val):
    dict_attr={}
    dict_attr['wrap_attr_name'] = type(val).__name__
    if val is not None:
        try:
            for prop,value in val.__dict__.items():
                dict_attr[prop]=value
        except Exception as e:
            print('erro while creating object for ',e)
    return dict_attr

def recognized_model_attr(value):
    att_cls = value['wrap_attr_name']
    try:
        model_attr= recognized_model_attr_map[att_cls]()
        for key in value.keys():
            print(key)
            key_type=type(value[key]);
            if key_type==list:
                setattr(model_attr,key,np.array(value[key]))
            else:
                setattr(model_attr,key,value[key])
        return model_attr
    except:
        print('Error in loading attributes for ', att_cls)
    return None

def get(dict_model):
    print(dict_model)
    dict_model = pd.read_json(dict_model,typ='series')
    model_type=dict_model['wrap_model_name']
    saved_model= recognized_models[model_type]()
    for key in dict_model.keys():
        key_type=type(dict_model[key])
        print(key, key_type)
        if key_type==list:
            setattr(saved_model,key,np.array(dict_model[key]))
        elif key_type.__name__ in std_type:
            setattr(saved_model,key,dict_model[key])
        else:
            setattr(saved_model,key,recognized_model_attr(dict_model[key]))
    return saved_model