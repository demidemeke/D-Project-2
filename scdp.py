import streamlit as st
import pandas as pd
#import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

st.write("""

""")

st.write("""
    
    # Predicting Sand Casting Surface Defects using a Data-Driven Supervised Machine Learning Approach: A Case Study at Akaki Basic Metals Industry

    ## Surface Cast Defect Prediction App

    ## This app predicts the **Surface Cast Defect Type** based on various features.
    
    **By: Demewez Demeke (GSR/0439/15)**

    **Advisor: Dr. Mesfin Gizaw**

    **Co-Advisor: Mr. Henok Zewdu (Ph.D. Candidate)**

    In the detection of advancing quality control practices within the global metal casting industry, the research focuses on the Akaki Basic Metals Foundry in Ethiopia, a key player facing challenges associated with surface defects in cast iron and steel metal components. Utilizing data-driven machine learning, particularly the supervise algorithm, the study aims to predict and prevent surface defects, mitigating structural integrity issues and reducing environmental impact. The research methodology involves a comprehensive literature survey, historical data collection, exploratory analysis, feature engineering, and model optimization. The anticipated outcome is a predictive model not only enhancing defect identification but also contributing to energy minimization and sustainability goals. This initiative positions Akaki Basic Metals Foundry as a leader in quality assurance, aligning with global imperatives of economic viability, competitiveness, and environmentally responsible manufacturing practices.

    """)

# Dataset Information and How to Use
st.write("""

## Dataset Information

This dataset is collected manually by the researcher from Akaki Basic Metals Industry.
""")

# Load the Boston housing dataset from the original source
data_url = "./modified_dataset_for_shape.csv"
data = pd.read_csv(data_url)

# Separate features and target variable
X = data.drop(columns=['DT'])
Y = data['DT']

# Display first 5 elements of the dataset
st.write("## First 5 Elements of the Dataset")
st.write(X.head())

# Calculate correlation between features and targets
correlation = X.join(Y).corr()

# Plot histogram for correlation
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("### Correlation between Features and Target")
plt.figure(figsize=(18, 14))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
st.pyplot()


# Define your algorithms
algorithms = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_jobs=-1, random_state=42)
}


# Sidebar

# Sidebar for model selection
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox("Which model do you want?", list(algorithms.keys()))

# Header of Specify Input Parameters
st.sidebar.header('Select Input Parameter Values')

def user_input_features():
    MFoC = st.sidebar.slider('MFoC', float(X.MFoC.min()), float(X.MFoC.max()), float(X.MFoC.mean()))
    MT = st.sidebar.slider('MT', float(X.MT.min()), float(X.MT.max()), float(X.MT.mean()))
    OQ = st.sidebar.slider('OQ', float(X.OQ.min()), float(X.OQ.max()), float(X.OQ.mean()))
    ND = st.sidebar.slider('ND', float(X.ND.min()), float(X.ND.max()), float(X.ND.mean()))
    D = st.sidebar.slider('D', float(X.D.min()), float(X.D.max()), float(X.D.mean()))
    C = st.sidebar.slider('C', float(X.C.min()), float(X.C.max()), float(X.C.mean()))
    Si = st.sidebar.slider('Si', float(X.Si.min()), float(X.Si.max()), float(X.Si.mean()))
    Mn = st.sidebar.slider('Mn', float(X.Mn.min()), float(X.Mn.max()), float(X.Mn.mean()))
    S = st.sidebar.slider('S', float(X.S.min()), float(X.S.max()), float(X.S.mean()))
    Sn = st.sidebar.slider('Sn', float(X.Sn.min()), float(X.Sn.max()), float(X.Sn.mean()))
    GS = st.sidebar.slider('GS', float(X.GS.min()), float(X.GS.max()), float(X.GS.mean()))
    BC = st.sidebar.slider('BC', float(X.BC.min()), float(X.BC.max()), float(X.BC.mean()))
    GP = st.sidebar.slider('GP', float(X.GP.min()), float(X.GP.max()), float(X.GP.mean()))
    CaT = st.sidebar.slider('CaT', float(X.CaT.min()), float(X.CaT.max()), float(X.CaT.mean()))
    CoR = st.sidebar.slider('CoR', float(X.CoR.min()), float(X.CoR.max()), float(X.CoR.mean()))
    CTe = st.sidebar.slider('CTe', float(X.CTe.min()), float(X.CTe.max()), float(X.CTe.mean()))
    PR = st.sidebar.slider('PR', float(X.PR.min()), float(X.PR.max()), float(X.PR.mean()))
    PT = st.sidebar.slider('PT', float(X.PT.min()), float(X.PT.max()), float(X.PT.mean()))
    SoT = st.sidebar.slider('SoT', float(X.SoT.min()), float(X.SoT.max()), float(X.SoT.mean()))
    SGSD = st.sidebar.slider('SGSD', float(X.SGSD.min()), float(X.SGSD.max()), float(X.SGSD.mean()))
    ST = st.sidebar.slider('ST', float(X.ST.min()), float(X.ST.max()), float(X.ST.mean()))
    SF = st.sidebar.slider('SF', float(X.SF.min()), float(X.SF.max()), float(X.SF.mean()))
    CQ = st.sidebar.slider('CQ', float(X.CQ.min()), float(X.CQ.max()), float(X.CQ.mean()))
    SMC = st.sidebar.slider('SMC', float(X.SMC.min()), float(X.SMC.max()), float(X.SMC.mean()))
    AMP = st.sidebar.slider('AMP', float(X.AMP.min()), float(X.AMP.max()), float(X.AMP.mean()))
    ShT = st.sidebar.slider('ShT', float(X.ShT.min()), float(X.ShT.max()), float(X.ShT.mean()))
    MS = st.sidebar.slider('MS', float(X.MS.min()), float(X.MS.max()), float(X.MS.mean()))
    SoD = st.sidebar.slider('SoD', float(X.SoD.min()), float(X.SoD.max()), float(X.SoD.mean()))

    data = {
        'MFoC': MFoC,
        'MT': MT,
        'OQ': OQ,
        'ND': ND,
        'D': D,
        'C': C,
        'Si': Si,
        'Mn': Mn,
        'S': S,
        'Sn': Sn,
        'GS': GS,
        'BC': BC,
        'GP': GP,
        'CaT': CaT,
        'CoR': CoR,
        'CTe': CTe,
        'PR': PR,
        'PT': PT,
        'SoT': SoT,
        'SGSD': SGSD,
        'ST': ST,
        'SF': SF,
        'CQ': CQ,
        'SMC': SMC,
        'AMP': AMP,
        'ShT': ShT,
        'MS': MS,
        'SoD': SoD
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Main Panel

# Print specified input parameters
st.header('Selected Input parameters')
st.write(df)
st.write('---')




# Build Regression Model
#model = RandomForestRegressor()
#model = XGBClassifier()
#model.fit(X, Y)
# Apply Model to Make Prediction
#prediction = model.predict(df)

# Function to train and explain the selected model
def run_model(model_name):
    model = algorithms[model_name]
    model.fit(X, Y)
    
    prediction = model.predict(df)
    
    st.header(f'Prediction of Defect Type using {model_name}')
    st.write("Here is the predicted sand surface defect type:")
    st.write(prediction)
    st.write('---')
    
    st.write("""
         
         #### Result Explanation 
        
         0 : Non-Defective
         
         1 : Porosity
         
         2 : Other-Defects
         
         3 : Shrinkage
         
         """)

run_model(selected_model)

# Explanation of each column
st.write("""
#### Explanation of Each Column

**MFoC**: 'Material-Form-on-Casting',

**MT**: 'Material-Type ', 

**OQ**: 'Order-Quantity',

**ND**: 'Non-Defective', 

**D**: 'Defective', 

**C**: 'C', 

**Si**: 'Si', 

**Mn**: 'Mn', 

**Cr**: 'Cr', 

**P**: 'P', 

**S**: 'S', 

**Cu**: 'Cu',

**Sn**: 'Sn', 

**GS**: 'Grain-Size', 

**BC**: 'Binder-Content', 

**GP**: 'Gas-Permeability',

**CaT**: 'Casting-Time', 

**CoR**: 'Cooling-Rate', 

**CTe**: 'Casting-Temperature', 

**PR**: 'Pouring-Rate',

**PT**: 'Pouring-Temperature', 

**SoT**: 'Solidification -Time ',

**SGSD**: 'Sand-Grain-Size-Distribution', 

**sT**: 'Sand-Temperature', 

**SRR**: 'Sand-Reuse-Rate',

**SF**: 'Sand-Flowability', 

**CQ**: 'Core-Quality', 

**PQ**: 'Pattern-Quality', 

**MQ**: 'Mold-Quality',

**SMC**: 'Sand-Moisture-Content', 

**AMP**: 'Alloy-Melting-Point', 

**ShT**: 'Shakeout-Time',

**MS**: 'Mold-Strength', 

**Cmpt**: 'Compactability', 

**SiS**: 'Silica-Sand', 

**SoD**: 'Severeity-of-Defect'

**DT**: 'Defect-Type'

""")


st.write("""

## Thank you!

**Contributor:** Demewoz

""")