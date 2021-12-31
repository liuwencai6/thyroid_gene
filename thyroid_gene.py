import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler


#应用标题
st.title('The machine learning-based predictive model ')

# conf
st.sidebar.markdown('## Variables')

#Diameter_G = st.sidebar.selectbox('Diameter.G',('<5cm','5-10cm','>10cm'),index=0)
Age = st.sidebar.slider("Age", 1, 100, value=60, step=1)
diameter = st.sidebar.slider("Up and down diameter", 1, 100, value=30, step=1)
Echogenic_Foci = st.sidebar.selectbox("Echogenic Foci",(0,1,2,3))
Echogenicity = st.sidebar.selectbox("Echogenicity",(1,2,3))
Elasticity = st.sidebar.selectbox("Elasticity",(0,2,3,4,5))
Margin = st.sidebar.selectbox("Margin",(0,2,3))
Shape = st.sidebar.selectbox("Shape",(0,3))
Lymph_Nodes = st.sidebar.selectbox("Lymph Nodes",('No','Yes'),index=0)
#Grade = st.sidebar.selectbox("Grade",('Well differentiated','Moderately differentiated','Poorly differentiated',
#                                      'Undifferentiated; anaplastic','unknown'),index=0)
#T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4','TX'))
#M = st.sidebar.selectbox("M stage",('M0','M1'))
#N = st.sidebar.selectbox("N stage",('N0','N1','N2','NX'))
#Primary_Site = st.sidebar.selectbox("Primary Site",('C64.9-Kidney','C65.9-Renal pelvis'))
#Sequence_number = st.sidebar.selectbox("Sequence number",('One primary only','more'))
#Brain_metastases = st.sidebar.selectbox("Brain metastases",('No','Yes'),index=0)
#Liver_metastasis = st.sidebar.selectbox("Liver metastasis",('No','Yes'),index=0)
#Bone_metastases = st.sidebar.selectbox("Bone metastases",('No','Yes'),index=0)
#Tumor_Size = st.sidebar.slider("Tumor size", 1, 999, value=30, step=1)
#steatosis = st.sidebar.selectbox("Steatosis",('No','Yes'),index=0)

#Lung_metastases = st.sidebar.selectbox("Lung metastases",('No','Yes'))
#st.sidebar.markdown('#  ')
# str_to_int

#map = {'T1':0,'T2':1,'T3':2,'T4':3,'TX':4,'No':0,'Yes':1,'Well differentiated':0,'Moderately differentiated':1,
#       'Poorly differentiated':2,'Undifferentiated; anaplastic':3,'unknown':4,'N0':0,'N1':1,'N2':2,'NX':3,
#       'C64.9-Kidney':0,'C65.9-Renal pelvis':1,'M0':0,'M1':1,'One primary only':0,'more':1}
map = {'No':0,'Yes':1}
Lymph_Nodes =map[Lymph_Nodes]

#Grade =map[Grade]
#T =map[T]
#N =map[N]
#Primary_Site =map[Primary_Site]
#Sequence_number =map[Sequence_number]
#Brain_metastases=map[Brain_metastases]
#Liver_metastasis =map[Liver_metastasis]
#Bone_metastases=map[Bone_metastases]
#Bone_metastases =map[Bone_metastases]
#Lung_metastases =map[Lung_metastases]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
features = [ 'Age', 'Up.and.down.diameter  ', 'Echogenicity', 'Shape','Margin', 'Echogenic.Foci', 'Elasticity', 'Lymph.Nodes' ]
target = 'Gene'
#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=32)
XGB.fit(X_ros, y_ros)
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=4,criterion='entropy',max_features='log2',max_depth=3,random_state=12)
#RF.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[Age, diameter, Echogenicity, Shape,Margin, Echogenic_Foci, Elasticity, Lymph_Nodes]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Age, diameter, Echogenicity, Shape,Margin, Echogenic_Foci, Elasticity, Lymph_Nodes]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')







