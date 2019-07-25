#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Python library to secure learned models from attacks:
## 1. Archtitectural/Network attacks
## 2. Adversarial attacks

## Assumptions
## 1. Single session per user
## 2. models used are from sklearn

## moves to REST API

import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.neural_network._stochastic_optimizers import *
import random

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
    #print('att_cls ',att_cls )
    try:
        model_attr= recognized_model_attr_map[att_cls]()
        for key in value.keys():
            #print(key)
            key_type=type(value[key]);
            if key_type==list:
                setattr(model_attr,key,np.array(value[key]))
            else:
                setattr(model_attr,key,value[key])
        return model_attr
    except:
        #print('Error in loading attributes for ', att_cls)
        return None


def save(model):
    model_type = type(model).__name__
    dict_attr={}
    dict_attr['wrap_model_name'] = model_type
    for prop,val in model.__dict__.items():
        key_type=type(val)
        #print(prop, key_type,key_type.__name__)
        if key_type.__name__ in std_type:
            dict_attr[prop]=val
        else:
            dict_attr[prop]=getClasAtr(val)
    model_key=''
    model_key=post_req(pd.Series(dict_attr).to_json())
    #print(dict_attr)
    return model_key

def get(model_key):    
    dict_model = get_req(model_key)
    #print(dict_model)
    dict_model = pd.read_json(dict_model,typ='series')
    model_type=dict_model['wrap_model_name']
    saved_model= recognized_models[model_type]()
    for key in dict_model.keys():
        key_type=type(dict_model[key])
        #print(key, key_type)
        if key_type==list:
            setattr(saved_model,key,np.array(dict_model[key]))
        elif key_type.__name__ in std_type:
            setattr(saved_model,key,dict_model[key])
        else:
            setattr(saved_model,key,recognized_model_attr(dict_model[key]))
    return saved_model

def verify(model_id,data,res):
    return verify_req(model_id,data,res)

import json
import requests


import requests

def post_req(data_model):
    headers = {'charset': 'utf-8'}
    url = 'http://localhost:8090/save'
    response = requests.post(url, json=data_model, headers=headers)
    print(response.status_code)
    return response.json()['refID']
    
def get_req(model_id):
    headers = {'charset': 'utf-8'}
    url = 'http://localhost:8090/get'
    req_params={'refID':model_id}
    response = requests.get(url,headers=headers,params = req_params)
    print(response.status_code)
    return response.json()['model']

def verify_req(model_id,data,res):
    headers = {'charset': 'utf-8'}
    url = 'http://localhost:8090/verify'
    req_params={}
    req_params['refID']=model_id
    req_params['test_data']=pd.Series(data).to_json(orient='values')
    req_params['test_res']=str(res)
    #print(req_params)
    response = requests.get(url,headers=headers,json = req_params)
    print(response.status_code)
    return response.json()['isCorrupted']


# In[ ]:




