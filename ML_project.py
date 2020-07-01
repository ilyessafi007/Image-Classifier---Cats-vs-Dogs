#import
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
import glob
import cv2
from PIL import Image
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,accuracy_score,mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg

#1 télécharger la base en python.

path_cats=glob.glob('Animals/cat*')
path_dogs=glob.glob('Animals/dog*')

#2 augmenter la base de données.
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


for image in path_cats:
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='Animals', save_prefix='cat', save_format='jpg'):
        i += 1
        if i > 60:
            break
for image in path_dogs:
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='Animals', save_prefix='dog', save_format='jpg'):
        i += 1
        if i > 60:
            break

#3 Transformer les images du niveau RGB vers niveau noir blanc.
path_cats=glob.glob('Animals/cat*')
path_dogs=glob.glob('Animals/dog*')

for im in path_cats:
    image = cv2.imread(im)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(im, image_gray)

for im in path_dogs:
    image = cv2.imread(im)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(im, image_gray)

#4 Rendre toutes les images de même taille afin de les utiliser pour la classification.
for im in path_cats:
    img = Image.open(im)
    img = img.resize((350,350))
    img.save(im)

for im in path_dogs:
    img = Image.open(im)
    img = img.resize((350,350))
    img.save(im)

#5 Transformer les images traitées en des matrices, puis en des vecteurs.
c=0
d=0
Data=[]
for im in path_cats:
    im1=Image.open(im)
    img = np.array(im1)
    img = img.reshape(img.shape[0]*img.shape[1])
    Data.append(img)
    c+=1
for im in path_dogs:
    im1=Image.open(im)
    img = np.array(im1)
    img = img.reshape(img.shape[0]*img.shape[1])
    Data.append(img)
    d+=1
#Labels
y=[]
for i in range(c):
    y.append(0)
for i in range(d):
    y.append(1)

c = list(zip(Data, y))
random.shuffle(c)
Data, y = zip(*c)
Data=list(Data)
y=list(y)

#6 algorithme pour la réduction des dimensions: PCA

pca=PCA(n_components=2200)
X1=pca.fit_transform(Data)

#7 l'algorithme adéquate pour la discrimination entre les chiens et les chats

#7.1 SVM Polynomial

X1_train, X1_test, y_train, y_test= train_test_split(X1,y, test_size=0.3, random_state=0)
svc_model_poly = svm.SVC(kernel='poly',degree=5)
svc_model_poly.fit(X1_train, y_train)
predictions_poly = svc_model_poly.predict(X1_test)
print("nbr of features: ", 2200, " PCA accuracy with POLY SVM " + str(100 * accuracy_score(y_test, predictions_poly))+'%')

#7.2 Random Forest

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X1_train, y_train)
y_pred=rf.predict(X1_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred.round())*100)
classification_report(y_test, y_pred.round())
#7.3 XGBOOST

data_dmatrix = xg.DMatrix(data=X1,label=y)
xg_reg = xg.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 20, alpha = 150, n_estimators = 1000)
xg_reg.fit(X1_train,y_train)
preds = xg_reg.predict(X1_test)
print("Accuracy:",metrics.accuracy_score(y_test, preds.round())*100)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


