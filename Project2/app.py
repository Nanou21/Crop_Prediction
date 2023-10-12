
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import sklearn
app = Flask(__name__)


#load model
model = pickle.load(open('model.pkl', 'rb'))


# In[5]:

@app.route("/")
def Home():
    return render_template('index.html')


# In[6]:


@app.route('/predict', methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template('index.html', prediction_text = "The crop is a {}".format(prediction))


# In[ ]:


if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
    app.run(debug = True)


# In[ ]:





# In[ ]:




