import numpy as np
import pickle
from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('./Data/fraudTrain.csv',index_col=0)
test_df = pd.read_csv('./Data/fraudTest.csv',index_col=0)
df = pd.concat([train_df,test_df])
df = df.drop(['trans_date_trans_time','dob','first','last','lat','long','trans_num','street','cc_num','merch_lat','merch_long','unix_time','merchant'],axis='columns')
df.select_dtypes(exclude=['float64','int64']).columns
df['category'] = df['category'].replace(',','')
df['gender'] = df['gender'].replace(',','')
df['city'] = df['city'].replace(',','')
df['state'] = df['state'].replace(',','')
df['job'] = df['job'].replace(',','')

col = ['category', 'gender', 'city', 'state', 'job']
categoryle = LabelEncoder()
genderle = LabelEncoder()
cityle = LabelEncoder()
statele = LabelEncoder()
joble = LabelEncoder()

df['category'] =  categoryle.fit_transform(df['category'])
df['gender'] =  genderle.fit_transform(df['gender'])
df['city'] =  cityle.fit_transform(df['city'])
df['state'] =  statele.fit_transform(df['state'])
df['job'] =  joble.fit_transform(df['job'])

# Loading ML model
model = pickle.load(open('./Models/best_model.sav', 'rb'))

# Create application
app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/showclf')
def showclf():
    return render_template('Classifier.html')


@app.route('/mlprediction', methods=['POST'])
def mlprediction():

    #features = [float(i) for i in request.form.values()]
    features = [i for i in request.form.values()]

    col = ['category', 'amt', 'gender', 'city', 'state', 'zip', 'city_pop', 'job']
    output_data=pd.DataFrame([features],columns = col)
    output_data['category'] = output_data['category'].replace(',','')
    output_data['gender'] = output_data['gender'].replace(',','')
    output_data['city'] = output_data['city'].replace(',','')
    output_data['state'] = output_data['state'].replace(',','')
    output_data['job'] = output_data['job'].replace(',','')

    output_data['amt'] = output_data['amt'].astype('float')
    output_data['zip'] = output_data['zip'].astype('int')
    output_data['city_pop'] = output_data['city_pop'].astype('int')
    
    output_data['category'] =  categoryle.transform(output_data['category'])
    output_data['gender'] =  genderle.transform(output_data['gender'])
    output_data['city'] =  cityle.transform(output_data['city'])
    output_data['state'] =  statele.transform(output_data['state'])
    output_data['job'] =  joble.transform(output_data['job'])
    print(output_data)

    prediction = model.predict(output_data)

    output = prediction
    print(output)


    if output == 1:
        return render_template('Classifier.html',
                               result='CREDIT CARD FRAUD DETECTED: ', positive='Yes', res2='Risk is HIGH')
    else:
        return render_template('Classifier.html',
                               result='CREDIT CARD FRAUD DETECTED: ', positive='No', res2='Risk is LOW')


if __name__ == '__main__':
    # Run the application
    app.run(debug=True)

