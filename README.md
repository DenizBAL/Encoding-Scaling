# Feature Engineering (Encoding & Scaling)
Bu bölümde Titanic verisi üzerinde:
- LabelEncoder ve OneHotEncoder ile kategorik veriler dönüştürülmüştür.
- MinMaxScaler kullanılarak sayısal veriler standartlaştırılmıştır.
- Korelasyon haritası ile değişken ilişkileri analiz edilmiştir.


import pandas as pd                     #KÜTÜPHANELER
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

df=pd.read_csv("titanic.csv")             #VERİ OKUMA
df.head()

 
df.isna().sum()                                                                                 #HATALI VERİ DÜZENLEME
df["age"].fillna(df["age"].median(), inplace=True)            
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["fare"].fillna(df["fare"].mean(),inplace=True)
 
le=LabelEncoder()                                                                                #LABEL ENCODİNG
df["sex_encoded"]=le.fit_transform(df["sex"])
df[["sex","sex_encoded"]].head()

df["embarked_mapped"]=df["embarked"].map({"C":0,"Q":1,"S":2})                                     ##MANUAL MAP ENCODİNG
df[["embarked","embarked_mapped"]].head(10)

embarked_dummies=pd.get_dummies(df["embarked"], prefix="embarked")                                #ON HİT ENCODİNG 
df=pd.concat([df,embarked_dummies], axis=1)      #OLUŞTURDUĞUMUZ YENİ SUTÜNLARI YATAYDA EKLİYORUZ.
df.head()

scaler=MinMaxScaler()                                                                             #BÜYÜK SAYILI VERİLERİ OKURKEN HATA YAPMASIN DİYE VERİYİ 0-1 ARASINDA SIKIŞTIRDIK.
df[["age_scaled","fare_scaled"]]=scaler.fit_transform(df[["age","fare"]])
df[["age_scaled","fare_scaled"]].head()

df_encoded=df[["sex_encoded","embarked_mapped","age_scaled","fare_scaled","survived"]]             #ENCODİNG YAPTIGIMIZ TÜM VERİLERİ TOPLUYORUZ.
df_encoded.head()

plt.figure(figsize=(8,5))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Kolerasyon Isı Haritası (Encoded+Scaled veriler)")
plt.show()
