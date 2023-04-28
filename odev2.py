# 1- Gerekli/Gereksiz bağımsız değişkenleri bulun.
# 2- 5 farklı yönteme göre regresyon modellerini çıkarınız.(MLR,PR,SVR,DT,RF)
# 3- Yöntemlerin başarılarını karşılaştırınız.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler = pd.read_csv("maaslar_yeni.csv")

x = veriler.iloc[:,2:3] # Bağımsız Değişkenlerimizi alır.(UnvanSeviyesi,Kidem,Puan)
y = veriler.iloc[:,5:] # Bağımlı Değişkeni alır.(Maas)

X = x.values
Y = y.values

# LİNEAR REGRESSİON (Tek değişken olma durumu ve doğrusal giden)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# OLS DEĞERİ HESAPLAMA
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

# POLYNOMİAL REGRESSİON (bağımsız değişken x ile bağımlı değişken y arasındaki ilişkinin x'te n'inci dereceden bir polinom olarak modellenmesi)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

model = sm.OLS(lin_reg2.predict(x_poly),X)
print(model.fit().summary())

# VERİLERİN ÖLÇEKLENMESİ

from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# SVM REGRESSİON (Değer ne çıkarsa çıksın önceden belirlenen kesin verilerden hangi değer aralığına girerse onu çıkarır.)
from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="black")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color ="green")
plt.show()

model = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model.fit().summary())

# DECİSİON TREE REGRESSİON (Bağımsız değişkenleri bilgi kazancına göre aralıklara ayırır.)

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor()
r_dt.fit(X,Y)

model = sm.OLS(r_dt.predict(X),X)
print(model.fit().summary())

# RANDOM FOREST REGRESSİON (Birden fazla karar ağacını kullanarak daha uyumlu modeller üreterek isabetli tahminlerde bulunur.)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

model = sm.OLS(rf_reg.predict(X),X)
print(model.fit().summary())


# KORELASYON ile veriler arasındaki değerler karşılaştırılarak daha iyi bir analiz yapılabilir.

print(veriler.corr())

#                Calisan ID  UnvanSeviyesi     Kidem      Puan      maas
# Calisan ID       1.000000       0.331847  0.206278 -0.251278  0.226287
# UnvanSeviyesi    0.331847       1.000000 -0.125200  0.034948  0.727036
# Kidem            0.206278      -0.125200  1.000000  0.322796  0.117964
# Puan            -0.251278       0.034948  0.322796  1.000000  0.201474
# maas             0.226287       0.727036  0.117964  0.201474  1.000000
