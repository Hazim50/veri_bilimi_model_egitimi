import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

class MyModel:
    def __init__(self, _train_path, _test_path, _model_input_path, _model_output_path, _random_state, _use_split, _test_size):
        self.train_path=_train_path
        self.test_path=_test_path
        self.model_input_path=_model_input_path
        self.model_output_path=_model_output_path
        self.random_state=_random_state
        self.use_split=_use_split
        self.test_size=_test_size

        self.train_df=None
        self.test_df=None
        self.frequency_table={}
        self.matching_table={}
        self.model=None
        self.X_test=None
        self.y_test=None

    def analyse_data(self):
        num_data=len(self.train_df)
        print(f"Toplam Kayıt Sayısı: {num_data}\n")

        print("Sütun İsimleri ve Nitelik Sayıları: ")
        for column in self.train_df.columns:
            num_attribute = self.train_df[column].nunique()
            most_freq_value=self.train_df[column].mode()[0]
            print(f"{column}: {num_attribute} adet nitelik var, en çok tekrar eden veri: {most_freq_value}")

        print(f"Sütun Nitelik Tipleri:\n{self.train_df.dtypes}")

    def plot(self):
        plt.figure(figsize=(20,15))
        for i, sutun in enumerate(self.train_df.columns, 1):
            plt.subplot(4, 5, i)
            self.train_df[sutun].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f'{sutun} Sütun Grafiği')
            plt.xlabel(sutun)
            plt.ylabel('Frekans')
        plt.tight_layout()
        plt.show()

    def read_xlsx(self):
        self.train_df=pd.read_excel(self.train_path)
        self.test_df=pd.read_excel(self.test_path)

    def create_frequency_table(self):
        for column in self.train_df.columns:
            most_freq_value=self.train_df[column].mode()[0]
            self.frequency_table[column]=most_freq_value

    def fill_missing_values(self):
        for column in self.train_df:
            self.train_df[column].fillna(self.frequency_table.get(column), inplace=True)

        for column in self.test_df:
            self.test_df[column].fillna(self.frequency_table.get(column), inplace=True)

    def create_matching_table(self):
        for column in self.train_df.columns:
            unique_items=self.train_df[column].unique()
            self.matching_table[column] = {deger: indeks for indeks, deger in enumerate(unique_items)}

    def apply_my_matching_system(self):
        for column in self.train_df.columns:
            if column in self.matching_table:
                self.train_df[column] = self.train_df[column].map(self.matching_table[column])

        for column in self.test_df.columns:
            if column in self.matching_table:
                self.test_df[column] = self.test_df[column].map(self.matching_table[column])

    def train(self):
        if self.use_split:
            dataset_columns = self.train_df.columns.difference(['Fiyat']).tolist()
            X_train, X_test, y_train, y_test = train_test_split(self.train_df[dataset_columns], self.train_df['Fiyat'],test_size=self.test_size, random_state=self.random_state)
            self.X_test,self.y_test=X_test,y_test
        else:
            dataset_columns = self.train_df.columns.difference(['Fiyat']).tolist()
            X_train, y_train = self.train_df[dataset_columns], self.train_df['Fiyat']

        self.model = RandomForestClassifier(random_state=self.random_state)
        self.model.fit(X_train, y_train)

    def predict(self):
        if self.use_split:
            X_test, y_test = self.X_test, self.y_test
        else:
            dataset_columns = self.test_df.columns.difference(['Fiyat']).tolist()
            X_test, y_test = self.test_df[dataset_columns], self.test_df['Fiyat']
        
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, zero_division=1)
        print(f'Accuracy: {accuracy}')
        print('\nClassification Report:\n', classification_rep)

    def save_model(self):
        with open(self.model_output_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self):
        with open(self.model_input_path, 'rb') as file:
            self.model = pickle.load(file)





if __name__=="__main__":
    train_path="C:\\Users\\Hazim\\Downloads\\data.xlsx"
    test_path="C:\\Users\\Hazim\\Downloads\\test.xlsx"
    model_input_path="C:\\Users\\Hazim\\Downloads\\model.pkl"
    model_output_path="C:\\Users\\Hazim\\Downloads\\model.pkl"
    random_state=11
    use_split=True
    test_size=0.2

    model=MyModel(train_path, test_path, model_input_path, model_output_path, random_state,use_split, test_size)
    model.read_xlsx()
    # model.analyse_data()
    model.create_frequency_table()
    model.fill_missing_values()
    # model.plot()
    model.create_matching_table()
    model.apply_my_matching_system()
    model.train()
    model.predict()
    # model.save_model()
    # model.load_model()
