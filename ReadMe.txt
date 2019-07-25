WRAP API

REST API to save and access trained models with version control

Required
 - Python 3
 - sklearn
 - tensorflow

1. All the python scripts to be extracted to a folder, /csec/project/
2. Modify and replace the path /csec/project/ on python scripts to complete path on your system
3. Create a directory /csec/project/wrapped/private/ for saving encrypted files.
4. Run following command to start WRAP server:
    $ python wrap_server.py
5. Import wrap_client in you project and call save(your_model) to get the model key and save ot on server as private service
6. Sample reference wrap.py with Logistic Regression and MLP classifier with simulated network attacks
7. Also included test data used for these models