from flask import Flask,request,send_from_directory,render_template
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os

app=Flask(__name__)

# STATIC_FOLDER='static'
STATIC_FOLDER='G:/Code/py prog/machine learning-deep learning/my projects/covid-19 prediction/deployment/webapp/static'

UPLOAD_FOLDER=STATIC_FOLDER+'/uploads/' #path to the folder where we will store the upload before prediction
MODEL_FOLDER=STATIC_FOLDER+'/models/' #path to the folders where we'll store the models

def predict(fullpath):
    data=image.load_img(fullpath,target_size=(100,100,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/model.h5')
    result=model.predict(data)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result=predict(fullname)
        pred=np.argmax(result,axis=1)
        prob_str=result
        # prob_str=' '.join(map(str, np.round_(prob_str*100,decimals=2)))

        if pred==0:
            label='Healthy'
            accuracy=np.round_(np.amax(prob_str)*100,decimals=2)
            return render_template('predict.html',image_file=file.filename,prediction_text='There is a {}% chance that these are {} lungs.'.format(accuracy,label))

        elif pred==1:
            label='Viral Pneumonia'
            accuracy=np.round_(np.amax(prob_str)*100,decimals=2)
            return render_template('predict.html',image_file=file.filename,prediction_text='There is a {}% chance that these lungs have {}.'.format(accuracy,label))

        else:
            label='Covid-19'
            accuracy=np.round_(np.amax(prob_str)*100,decimals=2)
            return render_template('predict.html',image_file=file.filename,prediction_text='There is a {}% chance that these lungs are infected with {}.'.format(accuracy,label))

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

if __name__=='__main__':
    app.run(debug=True)
        
