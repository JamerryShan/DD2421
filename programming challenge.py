import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def preprocessing_data():

    # read in training dataset
    TOM = pd.read_csv('TrainOnMe-2.csv', index_col=False)
    TOM.drop_duplicates()
    TOM = TOM[pd.to_numeric(TOM['x6'], errors='coerce').notnull()]
    TOM['x6'] = TOM['x6'].astype(float)
    TOM = TOM.dropna(axis=0, how='any')
    TOM = TOM.drop(columns=['Unnamed: 0'])  # Drop index column in the csv file

    # read in evaluate dataset
    EOM = pd.read_csv('EvaluateOnMe-2.csv', index_col=False)
    EOM = EOM[pd.to_numeric(EOM['x6'], errors='coerce').notnull()]
    EOM['x6'] = EOM['x6'].astype(float)
    EOM = EOM.drop(columns=['Unnamed: 0'])  # Drop index column in the csv file

    # encode categorical columns
    feature_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    TOM['y'] = label_encoder.fit_transform(TOM['y']) # Encode labels as integers instead
    TOM['x7'] = feature_encoder.fit_transform(TOM['x7']) # Encode strings as integers instead
    TOM['x12'] = feature_encoder.fit_transform(TOM['x12'])  # Encode booleans as integers instead
    EOM['x7'] = feature_encoder.fit_transform(EOM['x7']) # Encode strings as integers instead
    EOM['x12'] = feature_encoder.fit_transform(EOM['x12'])  # Encode booleans as integers instead

    visualizing(TOM)
    TOM_feature = TOM.drop(columns=['y'],axis=1)
    TOM_label = TOM['y']
    # Remove outliers in training data
    cols = [i for i in TOM.columns if i not in ['y', 'x7', 'x12']] # Ignore categorical features
    for col in cols:
        TOM = TOM[((TOM[col] - TOM[col].mean()) / TOM[col].std()).abs() < 3]

    choose_feature(TOM_feature, EOM)
    return (TOM_label, TOM_feature, EOM, label_encoder)

def visualizing(data):
    df2 = pd.DataFrame.copy(data)
    df2['y'] = df2['y'].astype('category').cat.codes
    df2['x7'] = df2['x7'].astype('category').cat.codes
    df2['x12'] = df2['x12'].astype('category').cat.codes

    df_train = df2.iloc[0:1000]
    df_eval = df2.iloc[1000:]
    params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 15),
              'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    pd.plotting.scatter_matrix(df_train.drop(['y'], axis=1), c=df_train['y'], marker='.', alpha=0.2)
    plt.draw()
    plt.show()

def choose_feature(train, evaluate):
    f = ['x1','x2','x3','x4','x5','x7','x8','x10','x11','x12']
    train = train.loc[:, f]
    evaluate = evaluate.loc[:, f]
    return train, evaluate

def predict_on_evaluate(model, encoder, data):
    final_result = model.predict(data)
    f = encoder.inverse_transform(final_result)
    np.savetxt('Evaluate_label.txt', f, fmt='%s')

def main():
    label, feature, evaluate, encoder = preprocessing_data()

    # define some models
    lr = LogisticRegression(solver='liblinear')
    ABDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=1000, learning_rate=0.8)
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    LDA = LinearDiscriminantAnalysis(solver='lsqr')
    NB = GaussianNB()

    # Implementing cross validation to decide which model is the best
    model = rf
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, feature, label, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    scores = scores.mean()
    print("10-fold cv accuracy is:", scores )

    # model.fit(feature, label)
    # predict_on_evaluate(model, encoder, evaluate)

if __name__ == '__main__':
    main()