import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import lime.lime_tabular


### LOAD DATA
@st.cache(allow_output_mutation=True)
def load_data(path):
    data = pd.read_csv(path)
    return data


readable_data_preprocessed = load_data('data/outputs/readable_data_preprocessed.csv')
data_final = load_data('data/outputs/data_final.csv')

added_columns = ['NAME_CONTRACT_TYPE', 'APPROVED_APP_CREDIT_PERC_MEAN', 'NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE',
                 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR']
data_final['NAME_CONTRACT_TYPE'] = readable_data_preprocessed['NAME_CONTRACT_TYPE']
data_final['APPROVED_APP_CREDIT_PERC_MEAN'] = readable_data_preprocessed['APPROVED_APP_CREDIT_PERC_MEAN']
data_final['NAME_FAMILY_STATUS'] = readable_data_preprocessed['NAME_FAMILY_STATUS']
data_final['NAME_EDUCATION_TYPE'] = readable_data_preprocessed['NAME_EDUCATION_TYPE']
data_final['OCCUPATION_TYPE'] = readable_data_preprocessed['OCCUPATION_TYPE']
data_final['ORGANIZATION_TYPE'] = readable_data_preprocessed['ORGANIZATION_TYPE']
data_final['NAME_HOUSING_TYPE'] = readable_data_preprocessed['NAME_HOUSING_TYPE']
data_final['FLAG_OWN_CAR'] = readable_data_preprocessed['FLAG_OWN_CAR']



@st.cache
def load_pickles(path):
    obj = pickle.load(open(path, 'rb'))
    return obj


model = load_pickles('pickles/final_model.pkl')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


local_css("style.css")

### SIDEBAR
st.sidebar.header('Client Selection')


def select_client():
    id_client = st.sidebar.selectbox('ID Client', data_final['SK_ID_CURR'].astype(int).sort_values())
    df_mask = data_final[data_final['SK_ID_CURR'] == id_client]
    df_mask['TARGET_1_PROBA'] = data_final['TARGET_1_PROBA']
    idx_client = data_final[data_final['SK_ID_CURR'] == id_client].index[0]
    return id_client, df_mask, idx_client


id_client, df_mask, idx_client = select_client()

#### TITLE
st.title("Credit Scoring App")

st.write("This Web Based Application is based on a Machine Learning Algorithm that predicts a client's chance of "
         "repaying a Credit or not. [Note that this model is 76% accurate]")
st.write('________________________________________________')

if df_mask['TARGET'].values[0] == 0:
    t = "<div><span class='highlight green'>Low Risk</span></div>"
    st.markdown(t, unsafe_allow_html=True)
else:
    t = "<div><span class='highlight red'>High Risk</span></div>"
    st.markdown(t, unsafe_allow_html=True)
st.write('#')

col1, col2 = st.columns(2)
### LOAN INFOS
with col1:
    st.subheader('__Credit Details__')
    st.write('Credit Default Risk:', round(df_mask['TARGET_1_PROBA'].values[0], 2))
    st.write('Risk Category (0:OK/1:Risked):', df_mask['TARGET'].values[0])
    st.write('Contract:', df_mask['NAME_CONTRACT_TYPE'].values[0])
    st.write('Amount:', df_mask['AMT_CREDIT'].values[0])
    st.write('Annuity:', df_mask['AMT_ANNUITY'].values[0])
    st.write('Income Credit Ratio:', round(df_mask['INCOME_CREDIT_PERC'].values[0] * 100), '%')
    st.write('Annuity Income Ratio:', round(df_mask['ANNUITY_INCOME_PERC'].values[0] * 100), '%')
    st.write('Payment Rate:', round(df_mask['PAYMENT_RATE'].values[0] * 100), '%')
    st.write('Mean Approved Credit Ratio:', round(df_mask['APPROVED_APP_CREDIT_PERC_MEAN'].values[0], 2))

### CLIENT INFOS
with col2:
    st.subheader('__Client Infos__')

    st.write('Gender:', df_mask['CODE_GENDER'].values[0])
    st.write('Age:', round(df_mask['DAYS_BIRTH'].values[0] / (-365)))
    st.write('Status:', df_mask['NAME_FAMILY_STATUS'].values[0])
    st.write('Education:', df_mask['NAME_EDUCATION_TYPE'].values[0])
    st.write('Occupation:', df_mask['OCCUPATION_TYPE'].values[0])
    st.write(
        'Employer:', df_mask['ORGANIZATION_TYPE'].values[0],
        ', for', round(df_mask['DAYS_EMPLOYED'].values[0] / (-365)),
        'years')
    st.write('Days Employed Percentage:', round(df_mask['DAYS_EMPLOYED_PERC'].values[0] * 100), '%')
    st.write('Housing:', df_mask['NAME_HOUSING_TYPE'].values[0])
    st.write('Car:', df_mask['FLAG_OWN_CAR'].values[0])

data_final.drop(added_columns, axis=1, inplace=True)

# Imputer
data_final.replace([np.inf, -np.inf], np.nan, inplace=True)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
data_final = pd.DataFrame(imp.fit_transform(data_final), columns=data_final.columns)

st.write('________________________________________________')
st.subheader('Client Raw Data')
st.write(df_mask)

st.write('________________________________________________')
st.subheader('Client Comparison using k-NN')
n_neighbors = st.slider('Number of similar clients to find:', min_value=0, step=1)


def neighbors(i, n_neighbors):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data_final.drop('SK_ID_CURR', axis=1))
    neighbors_idx = nbrs.kneighbors(data_final.drop('SK_ID_CURR', axis=1).iloc[[i]], n_neighbors=n_neighbors + 1,
                                    return_distance=False)
    neighbors_mean = pd.DataFrame(data_final.iloc[neighbors_idx[0]].mean(axis=0)).T
    comparison = pd.DataFrame(data_final.iloc[i]).T.append(neighbors_mean).reset_index(drop=True)
    comparison.rename(index={0: 'Client', 1: 'Neighbors Mean'}, inplace=True)
    st.write('Showing', n_neighbors, 'similar clients:')
    st.write(data_final.iloc[neighbors_idx[0]])
    st.write('Mean of similar clients:')
    st.write(comparison)


if n_neighbors > 0:
    with st.spinner('Calculating...'):
        neighbors(idx_client, n_neighbors)

### LOCAL EXPLANATION
st.write('________________________________________________')
st.subheader('Client Score Explanation')

# If button activated
if st.button('Generate Score Explanation'):
    with st.spinner('Calculating...'):
        class_names = ['0: Low Risk', '1: Failure Risk']  # distinct classes from the target variable
        # Explainer
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(training_data=data_final.drop([
            'TARGET', 'SK_ID_CURR', 'TARGET_1_PROBA'], axis=1).values,  # Needs to be in Numpy array format
                                                                mode='classification',
                                                                training_labels=data_final['TARGET'],
                                                                feature_names=data_final.drop([
                                                                    'TARGET', 'SK_ID_CURR', 'TARGET_1_PROBA'],
                                                                    axis=1).columns,
                                                                class_names=class_names,
                                                                discretize_continuous=False)
        # Generate explanation
        explanation = explainer_lime.explain_instance(data_final.drop([
            'TARGET', 'SK_ID_CURR', 'TARGET_1_PROBA'], axis=1).loc[idx_client].values,
                                                      model.predict_proba,
                                                      num_features=data_final.drop(
                                                          ['TARGET', 'SK_ID_CURR', 'TARGET_1_PROBA'], axis=1).shape[1]
                                                      )
        # Plot explanation
        exp = explanation.as_list()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        pos = np.arange(len(vals))
        plot = plt.figure(figsize=(10, 10))
        plt.style.use('seaborn')
        colors = ['tab:red' if x > 0 else 'tab:green' for x in vals]
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        plt.title('Local explanation for class 1: Credit Default')
        st.pyplot(plot)

st.write('________________________________________________')
st.subheader('Overall Most Important Criterias')
st.image('data/outputs/lgbm_importances01.png')
