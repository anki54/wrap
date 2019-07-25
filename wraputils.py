#!/usr/bin/env python
# coding: utf-8

# In[17]:


## Python utilities used for aneryption/decryption of models by wrap
## Supprts functions to 
## 1. hash
## 2. encrypt(algoritm=)
## 3. decrypt(algorithm=)

## moves to REST API

from Crypto.Cipher import AES
import ctypes
import sys
sys.path.append("/csec/project/")
import json
from wrap_internal import *
from flask import jsonify

key='hr8300j9f8390839'
arg='This is an IV456'
private_dir='/csec/project/wrapped/private/'


def hash_key(any_object):
    return str(ctypes.c_size_t(hash(str(any_object))).value)

def encode(original_text):
    encryption_suite = AES.new(key, AES.MODE_CFB, arg)
    encoded_text = encryption_suite.encrypt(original_text)
    return encoded_text

def decode(encoded_text):
    encryption_suite = AES.new(key, AES.MODE_CFB, arg)
    original_text = encryption_suite.decrypt(encoded_text)
    return str(original_text,'utf-8')

def read_file(file_name):
    file_content=''
    model_file = open(private_dir+file_name+'.txt','rb')
    try:        
        encoded_file_content = model_file.read()
        file_content = decode(encoded_file_content)
        print(str(file_content))
    except Exception as e:
        print('failed to read model',e)
    model_file.close()
    return str(file_content)

def write_file(file_name,content):    
    print('write to file', file_name+'.txt')
    model_file = open(private_dir+file_name+'.txt','wb+')
    try:
        encoded_content = encode(str(content))
        model_file.write(encoded_content)
    except Exception as e:
        print('failed to save model',e)
    model_file.close()   

def predict(model_id,test_data):
    model = get(read_file(model_id))
    return model.predict(test_data)





