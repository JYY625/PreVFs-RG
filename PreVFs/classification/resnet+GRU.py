from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# from sklearn.externals import joblib
import utils.tools as utils
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
# from keras.layers import Dense,Input,Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D,ReLU,BatchNormalization,Activation

# 读取训练集
data_ = pd.read_csv(r"C:\Users\ACER\deepVF\DeepVF\DeepVF_Training_Dataset\all_train.csv", header=None)
data = np.array(data_)
np.random.seed(2022)
np.random.shuffle(data) # 打乱顺序
print(data.shape)
data = data[:, 0:]
[m1, n1] = np.shape(data)
# label = pd.read_csv(r"C:\Users\ACER\deepVF\DeepVF\DeepVF_Training_Dataset\DNN\DNN_label.csv",header=None)
label1 = np.ones((3000, 1))  # Value can be changed
label2 = np.zeros((4334, 1))
# 合并标签
label = np.append(label1, label2)
label = np.array(label)
np.random.seed(2022)
np.random.shuffle(label)
print(label.shape)
shu = scale(data)
X1 = shu
y = label

X = np.reshape(X1, (-1, 1, n1))
# X0 = np.reshape(X0, (-1, 1, n1))
# y0 = utils.to_categorical(y0)
inputs = Input(shape=(1,1282))
layer0 = Conv1D(filters=64, kernel_size=3, padding='same',activation='relu')(inputs)
# layer0 = tf.keras.layers.BatchNormalization(layer0)
# print(inputs)
# inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
layer1 = Conv1D(filters=128, kernel_size=3, padding='same',activation='relu')(layer0)
layer2 = Conv1D(filters=64, kernel_size=3, padding='same',activation='relu')(layer1)
skip0 = Activation("relu")(layer0)
skip0 = BatchNormalization()(skip0)
skip0 = Activation('relu')(skip0)
layer_temp1 = layer2+layer0
layer3 = layer_temp1
# layer3 = tf.convert_to_tensor(layer3,)
skip = Conv1D(filters=128, kernel_size=3,padding='same',activation='relu')(layer3)
layer4 = Conv1D(filters=256, kernel_size=3,padding='same',activation='relu')(layer3)
layer5 = Conv1D(filters=128, kernel_size=3, padding='same',activation='relu')(layer4)
skip = Conv1D(filters=128, kernel_size=1,padding='same',activation='relu')(layer3)
skip = BatchNormalization()(skip)
skip = Activation('relu')(skip)
layer6 = skip+layer5
# temp1 = Conv1D(filters=256, kernel_size=3,padding='same',activation='relu')(layer6)
# temp2 = Conv1D(filters=256, kernel_size=3, padding='same',activation='relu')(temp1)
# skip2 = Conv1D(filters=256, kernel_size=1, padding='same',activation='relu')(layer6)
# temp3 = temp2 + skip2
layer7 = GRU(128, return_sequences=True)(layer6)
# layer7 = Dropout(0.5)(layer7)
# layer8 = GRU(64, return_sequences=True)(layer7)
# layer8 = Dropout(0.5)(layer8)
# layer9 = GRU(16, return_sequences=True)(layer8)
# layer9 = Dropout(0.3)(layer9)
# layer10 = GRU(8, return_sequences=True)(layer9)
layer9 = Flatten()(layer7)
layer10 = Dense(64, activation='relu')(layer9)
layer11 = Dense(32, activation='relu')(layer10)
outputs = tf.keras.layers. Dense(2,activation="softmax")(layer11)

# model = tf.keras.Model(inputs,outputs)
# model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(X,y,epochs=50,validation_split=0.2)


sepscores = []

ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5

skf = StratifiedKFold(n_splits=10,shuffle=True)

for train, test in skf.split(X, y):
    tf.keras.backend.clear_session() #清除模型占用内存
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    # print(y[train].shape)
    # print('sum_y',np.sum(y[train]))
    # print('trainy',y[train][0:100])
    y_train = utils.to_categorical(y[train])  # generate the resonable results
    cv_clf = model
    hist = cv_clf.fit(X[train],
                      y_train, batch_size=30,
                      epochs=30)
    with open(r"C:\Users\ACER\deepVF\DeepVF\PreVFs-RG\cv_clf.pickle","preVFs")as f:
        pickle.dump(cv_clf,f)
    # acc1 = cv_clf.evaluate(X0,y0)
    # print('acc1',acc1)
    y_score = cv_clf.predict(X[test])  # the output of  probability
    y_class = utils.categorical_probas_to_classes(y_score)

    y_test = utils.to_categorical(y[test])  # generate the test
    ytest = np.vstack((ytest, y_test))
    y_test_tmp = y[test]
    yscore = np.vstack((yscore, y_score))

    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                        y_test_tmp)
    print('acc',acc)
    # fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    fpr6, tpr6, _ = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr6, tpr6)
    sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])

scores = np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))

result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscores.append(H1)
result = sepscores
row = yscore.shape[0]
yscore = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore)
# yscore_sum.to_csv(r'C:\Users\ACER\deepVF\DeepVF\DeepVF_Training_Dataset\plot\yscore_resnet+GRU.csv')
ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
# ytest_sum.to_csv(r'C:\Users\ACER\deepVF\DeepVF\DeepVF_Training_Dataset\plot\ytest_resnet+GRU.csv')
fpr6, tpr6, _ = roc_curve(ytest[:, 0], yscore[:, 0])
auc_score = np.mean(scores, axis=0)[7] * 100
lw = 2


# plt.plot(fpr6, tpr6, color='darkorange',
#          lw=lw, label='CNN+GRU ROC (auc_score = %0.2f%%)' % auc_score)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
# data_csv = pd.DataFrame(data=result)
# data_csv.to_csv(r'C:\Users\ACER\Desktop\deepVF\DeepVF\DeepVF_Training_Dataset\resnet+GRU.csv')




