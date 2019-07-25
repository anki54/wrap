#!/usr/bin/env python
# coding: utf-8

# In[13]:


## Build a Web Service that will save and secure models.
## APIs at work
## 1. save() --- save model config on server, return a ref ID, encrypt the model on server
## 2. get() --- return the saved model
## 3. verify() --- verifies the results against the encrypted model and return true/false model is corrupted or not.


from flask import Flask, jsonify, request
import sys
sys.path.append("/csec/project/")
import json
from wraputils import *

app = Flask(__name__)

@app.route('/save',methods=['POST'])
def save():
    model_id=None
    try:
        json_data = request.get_json(force=True)
        print(json_data)
        model_id=hash_key(json_data)
        print('model id ', model_id)
        write_file(model_id, json_data )
    except Exception as e:
        print(e)
        model_id=None
    return jsonify({'refID' : model_id})

@app.route('/get', methods=['GET'])
def get():
    args = request.args
    model_id = args['refID']
    print('model id ', model_id)
    model=read_file(model_id)
    return jsonify({'model' : model})

@app.route('/verify', methods=['GET'])
def verify():
    isCorrupted=False
    json_data = request.get_json(force=True)
    model_id = json_data['refID']
    test_data=json_data['test_data']
    test_res=json_data['test_res']
    print('modelid', model_id)
    print('test_data',test_data)
    print('test_res',test_res)
    reshaped_array = pd.read_json(test_data,typ='series')
    model_res = predict(model_id,reshaped_array.values.reshape(1, -1))
    print('model_res',model_res[0])
    print('test_res',test_res[1])
    if test_res[1] != str(model_res[0]):
        isCorrupted = True
    return jsonify({'isCorrupted' : isCorrupted})

if __name__ == '__main__':
    app.run(debug=True,port=8090)


# In[ ]:




