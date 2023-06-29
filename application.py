from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle

application = Flask(__name__)
app = application

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Get the uploaded file from the request
        uploaded_file = request.files['file']

        # Save the file to the predictions folder
        filename = secure_filename(uploaded_file.filename)
        saved_path = f'predictions/{filename}'
        uploaded_file.save(saved_path)

        df = pd.read_csv(saved_path)
        f = open("cols_to_drop.csv", "r")
        drop_cols=f.readlines()
        s=str(drop_cols)
        cleaned_string = s.strip("[]'")

        # Split the cleaned string on commas
        split_list = cleaned_string.split(',')

        # Remove any leading or trailing spaces from each item in the split list
        drop_columns = [item.strip() for item in split_list]
        drop_columns = [element for element in drop_columns if element != '']
        df=df.drop(drop_columns,axis=1)


        with open('preprocessing.pkl', 'rb') as file:
            pmodel = pickle.load(file)

        # Access the KNNImputer component
        imputer_component = pmodel.named_steps['Imputer']

        # Access the StandardScaler component
        scaler_component = pmodel.named_steps['Scaler']

        # Preprocess the data
        df_trans = imputer_component.transform(df)
        df_trans = scaler_component.transform(df_trans)

        with open('fault_detection.pkl', 'rb') as file:
            model = pickle.load(file)

        ans_arr = model.predict(df_trans)
        count_zero = np.count_nonzero(ans_arr == 0)
        count_one = np.count_nonzero(ans_arr == 1)
        p_faulty = count_zero / (count_zero + count_one)

        if p_faulty > 0.5:
            result = "Wafer is faulty"
        else:
            result = "Wafer is not faulty"

        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
