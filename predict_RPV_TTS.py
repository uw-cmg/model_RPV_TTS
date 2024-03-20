import os
import pandas as pd
import numpy as np
import joblib
import dill
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator

def get_preds_ebars_domains(df_test):
    d = 'model_RPV_TTS'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))
    recal_params = pd.read_csv(os.path.join(d, 'recal_dict.csv'))

    features = df_features.columns.tolist()
    df_test = df_test[features]

    X = scaler.transform(df_test)

    # Make predictions
    preds = model.predict(X)

    # Get ebars and recalibrate them
    errs_list = list()
    a = recal_params['a'][0]
    b = recal_params['b'][0]
    c = recal_params['c'][0]
    for i, x in X.iterrows():
        preds_list = list()
        for pred in model.model.estimators_:
            preds_list.append(pred.predict(np.array(x).reshape(1, -1))[0])
        errs_list.append(np.std(preds_list))
    ebars = a * np.array(errs_list)**2 + b * np.array(errs_list) + c

    # Get domains
    with open(os.path.join(d, 'model.dill'), 'rb') as f:
        model_domain = dill.load(f)

    domains = model_domain.predict(X)

    return preds, ebars, domains


def make_predictions(df_test):

    # Process data
    X_train = pd.read_csv('model_RPV_TTS/X_train.csv')
    feature_names = X_train.columns.tolist()

    # Convert Product form encoding to numbers
    pf = df_test['Product Form']
    pf_0 = list()
    pf_1 = list()
    pf_2 = list()
    pf_3 = list()
    pf_4 = list()
    pf_5 = list()
    for i in pf:
        if i == 'F':
            pf_0.append(1)
            pf_1.append(0)
            pf_2.append(0)
            pf_3.append(0)
            pf_4.append(0)
            pf_5.append(0)
        elif i == 'HAZ':
            pf_0.append(0)
            pf_1.append(1)
            pf_2.append(0)
            pf_3.append(0)
            pf_4.append(0)
            pf_5.append(0)
        elif i == 'P':
            pf_0.append(0)
            pf_1.append(0)
            pf_2.append(1)
            pf_3.append(0)
            pf_4.append(0)
            pf_5.append(0)
        elif i == 'PCE':
            pf_0.append(0)
            pf_1.append(0)
            pf_2.append(0)
            pf_3.append(1)
            pf_4.append(0)
            pf_5.append(0)
        elif i == 'SRM':
            pf_0.append(0)
            pf_1.append(0)
            pf_2.append(0)
            pf_3.append(0)
            pf_4.append(1)
            pf_5.append(0)
        elif i == 'W':
            pf_0.append(0)
            pf_1.append(0)
            pf_2.append(0)
            pf_3.append(0)
            pf_4.append(0)
            pf_5.append(1)
        else:
            raise ValueError('Product form must be one of F, HAZ, P, PCE, SRM, W')
    
    df_test['Product Form_0'] = pf_0
    df_test['Product Form_1'] = pf_1
    df_test['Product Form_2'] = pf_2
    df_test['Product Form_3'] = pf_3
    df_test['Product Form_4'] = pf_4
    df_test['Product Form_5'] = pf_5

    del df_test['Product Form']

    # Check the data
    cols_in = df_test.columns.tolist()
    for c_in in cols_in:
        if c_in not in feature_names:
            print('Error with input feature', c_in)
            print('Input features should be', feature_names)
            break

    # Get the ML predicted values
    preds, ebars, domains = get_preds_ebars_domains(df_test)

    pred_dict = {'Predicted TTS (degC)': preds,
                 'Ebar TTS (degC)': ebars}

    for d in domains.columns.tolist():
        pred_dict[d] = domains[d]

    del pred_dict['y_pred']
    #del pred_dict['d_pred']
    del pred_dict['y_stdu_pred']
    del pred_dict['y_stdc_pred']

    for f in feature_names:
        pred_dict[f] = np.array(df_test[f]).ravel()

    return pd.DataFrame(pred_dict)