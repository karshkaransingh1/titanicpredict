import streamlit as st
import pandas as pd
import joblib

model = joblib.load('titanic_model.pkl')

st.title('titanic survival predictor')
st.write('enter passenger details to predict')

st.sidebar.header('passenger details')

pclass = st.sidebar.selectbox('ticket class (1 2 3)', [1, 2, 3])

sex = st.sidebar.selectbox('gender', ['male', 'female'])

age = st.sidebar.slider('age', 0, 100, 25)

sibsp = st.sidebar.number_input('siblings/spouses aboard', 0, 10, 0)
parch = st.sidebar.number_input('parents/children aboard', 0, 10, 0)

sex_encoded = 1 if sex == 'male' else 0

input_data = pd.DataFrame([[pclass, age, sibsp, parch, sex_encoded]], 
                          columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male'])

if st.button('predict survival'):
    prediction = model.predict(input_data)[0]
    # remember probs returns 2d array
    # [0][1] gets seconds element which is the case 1 probablity

    probs = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"result: **Survived**")
    else:
        st.error(f"result: **did not survived**")
    
    st.metric(label= 'survival probablity', value = f"{probs:.2%}")

