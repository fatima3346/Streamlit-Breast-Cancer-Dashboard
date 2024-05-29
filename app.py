from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





def getPCA(df):
    pca = PCA(n_components=3)
    result = pca.fit_transform(df.loc[:, df.columns != 'Type'])
    df['pca-1'] = result[:, 0]
    df['pca-2'] = result[:, 1]
    df['pca-3'] = result[:, 2]
    return df


def return_data(dataset):
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names, index=None)
    df['Type'] = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=1, test_size=0.2)
    return X_train, X_test, y_train, y_test, df, data.target_names


# Title
st.title("Classifiers in Action")

# Description
st.text("Breast canser dataset")

# sidebar
sideBar = st.sidebar
dataset = 'Breast Cancer'
classifier = sideBar.selectbox(
    'Which Classifier do you want to use?', ('SVM', 'KNN'))


X_train, X_test, y_train, y_test, df, classes = return_data(dataset)
st.dataframe(df.sample(n=5, random_state=1))
st.subheader("Classes")

for idx, value in enumerate(classes):
    st.text('{}: {}'.format(idx, value))

# 2-D PCA
df = getPCA(df)
fig = plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="pca-1", y="pca-2",
    hue="Type",
    palette=sns.color_palette("hls", len(classes)),
    data=df,
    legend="full"
)
plt.xlabel('PCA One')
plt.ylabel('PCA Two')
plt.title("2-D PCA Visualization")
st.pyplot(fig)

# 3-D PCA
fig2 = plt.figure(figsize=(16, 10)).gca(projection='3d')
fig2.scatter(
    xs=df["pca-1"],
    ys=df["pca-2"],
    zs=df["pca-3"],
    c=df["Type"],
)
fig2.set_xlabel('pca-one')
fig2.set_ylabel('pca-two')
fig2.set_zlabel('pca-three')
plt.title("3-D PCA Visualization")
st.pyplot(fig2.get_figure())


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names, index=None)
df['Type'] = data.target
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=1, test_size=0.2)

classes = data.target_names
st.dataframe(df.sample(n=5, random_state=1))
st.subheader("Classes")


for idx, value in enumerate(classes):
    st.text('{}: {}'.format(idx, value))


if classifier == 'SVM':
    c = st.sidebar.slider(label='Choose value of C',min_value=0.0001, max_value=10.0)
    model = SVC(C=c)
    model.fit(X_train, y_train)
    test_score = round(model.score(X_test, y_test), 2)
    train_score = round(model.score(X_train, y_train), 2)
    accuracy = test_score
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(
        y_test, y_pred, labels=classes).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=classes).round(2))
    # The below code will plot confusion matrix for SVM
    st.subheader("ROC Curve")
    plot_roc_curve(model, X_test, y_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif classifier == 'KNN':
    neighbors = st.sidebar.slider(
        label='Choose Number of Neighbors', min_value=1, max_value=20)
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    test_score = round(model.score(X_test, y_test), 2)
    train_score = round(model.score(X_train, y_train), 2)
    accuracy = test_score
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(
        y_test, y_pred, labels=classes).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=classes).round(2))
    # The below code will plot confusion matrix for KNN
    st.subheader("ROC Curve")
    plot_roc_curve(model, X_test, y_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

