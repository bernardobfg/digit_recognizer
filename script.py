import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

#setamos o seed para reprodução do experimento
np.random.seed(2)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#retiramos a informação do digito
x_train = df_train.drop(["label"], axis=1).values
#apesar do dataset ja estar no formato 28x28, o framework do keras espera que seja
#informado a terceira dimensão,portanto já redimensionamentos para 28x28x1.
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = df_test.values.reshape((df_test.shape[0], 28, 28, 1))

# utilizamos a função to_categorial do utils do keras para fazermos o one-hot-encoder da classe.
y_train = df_train["label"].values
y_train = np_utils.to_categorical(y_train)

#visualizando randomicamente algumas imagens
for i in range(0, 6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    plt.subplot(3,2,i+1)
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplots_adjust(top=1.4)
plt.show()