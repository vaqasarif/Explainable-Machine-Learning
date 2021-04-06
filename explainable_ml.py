import pandas as pd
import numpy as np
import webbrowser
from category_encoders import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import SmartExplainer
feat_dic ={'mpg':"Miles per galon",
           'cylinders':"Number of cylinders",
             'displacement':"Mileage",
             'horsepower':"Engine capacity",
             'Weight':"bodyweight",
             'acceleration':"Speed",
             'model year':"Make Year",
             'origin':"From where",
             }
df = pd.read_csv( 'auto-mpg.csv', sep= ',')

df=df.drop(['car name'],axis=1)
y_df=df['mpg'].to_frame()
X_df=df[df.columns.difference(['mpg'])]

from category_encoders import OrdinalEncoder

categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']

encoder = OrdinalEncoder(
    cols=categorical_features,
    handle_unknown='ignore',
    return_df=True).fit(X_df)

X_df=encoder.transform(X_df)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)


regressor = LGBMRegressor(n_estimators=200).fit(Xtrain,ytrain)

y_pred = regressor.predict(Xtest)

from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(ytest,y_pred)))

xpl = SmartExplainer(features_dict=feat_dic)

xpl.compile(
    x=Xtest,
    model=regressor,
    preprocessing=encoder # Optional: compile step can use inverse_transform method
)

app = xpl.run_app(title_story='Miles Per Galon prediction')

webbrowser.open('http://127.0.0.1:8050/')
