import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

DATA_DIR = os.path.join(os.path.abspath('..'), "data")
IMG_DIR = os.path.join(os.path.abspath('..'), "images")


class CICDdOS2019Preprocessor(object):
    def __init__(self, data_path, training_size, testing_size, validation_size):
        self.data_path = data_path
        self.training_size = training_size
        self.testing_size = testing_size
        self.validation_size = validation_size

        self.data = None
        self.features = None
        self.labels = None

    def read_data(self):
        filenames = glob.glob(os.path.join(self.data_path, 'raw', '*.csv'))
        datasets = [pd.read_csv(filename, low_memory=False) for filename in filenames]

        # xóa khoảng trắng và lower + replace các dấu /
        for dataset in datasets:
            dataset.columns = [self.clean_column_name(column) for column in dataset.columns]

        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        self.data.drop(columns=self.data.columns[0], axis=1, inplace=True)
        self.data.drop(labels=['fwd_header_length.1', 'unnamed:_0', 'flow_id', 'source_ip', 'destination_ip',
                               'timestamp', 'simillarhttp'],
                       axis=1,
                       inplace=True)  # cột này trùng => bỏ

    def clean_column_name(self, col):
        col = col.strip(' ')
        col = col.replace('/', '_')
        col = col.replace(' ', '_')
        col = col.lower()
        return col

    def remove_duplicate_values(self):
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_constant_features(self, threshold=0.01):
        # constant feature: chỉ có 1 giá trị cho tất cả các samples

        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.iteritems() if std < threshold]
        print(constant_features)

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        # correlated freature: nếu giữa 2 hoặc nhiều feature có sự tương quan cao
        # có nghĩa là ta có thể suy ra cái còn lại từ 1 cái đã cho, nghĩa là feature thứ 2
        # không mạng lại thêm thông tin gì cho việc dự đoán target => bỏ cái thứ 2 đi

        # Correlation matrix
        data_corr = self.data.corr()
        fig = plt.figure(figsize=(15, 15))
        sns.set(font_scale=1.0)
        ax = sns.heatmap(data_corr, annot=False)
        fig.savefig(os.path.join(IMG_DIR, 'correlation_matrix.pdf'))

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        print(correlated_features)

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        # Thêm label pair vào nếu có thêm loại tấn công nào trong tập dataset
        attack_group = {
            'BENIGN': 'Benign',
            'Portmap': 'Portmap',
            'DrDoS_DNS': 'DDos',
            'DrDoS_MSSQL': 'DDos',
            'DrDoS_NetBIOS': 'DDos',
            'DrDoS_NTP': 'DDos',
            'DrDoS_SNMP': 'DDos',
            'DrDoS_SSDP': 'DDos',
            'DrDoS_UDP': 'DDos',
            'DrDoS_LDAP': 'DDos',
            'Syn': 'Syn',
            'TFTP': 'TFTP',
            'UDP-lag': 'UDPLag',
            'WebDDoS': 'DDos'
        }

        # attack_group = {
        #     'BENIGN': 'Benign',
        #     'Portmap': 'Attack',
        #     'DrDoS_DNS': 'Attack',
        #     'DrDoS_MSSQL': 'Attack',
        #     'DrDoS_NetBIOS': 'Attack',
        #     'DrDoS_NTP': 'Attack',
        #     'DrDoS_SNMP': 'Attack',
        #     'DrDoS_SSDP': 'Attack',
        #     'Syn': 'Attack',
        #     'TFTP': 'Attack',
        #     'UDP-lag': 'Attack',
        #     'WebDDoS': 'Attack'
        # }

        # Tạo 1 cột label_category
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])

    def train_valid_test_split(self):
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)

        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.labels,
                                                            test_size=(self.testing_size + self.validation_size),
                                                            random_state=42, shuffle=True)
        X_test, X_val, Y_test, Y_val = train_test_split(
            X_test,
            Y_test,
            test_size=self.validation_size / (self.validation_size + self.testing_size),
            random_state=42,
            shuffle=True
        )

        return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
        # return (X_train, Y_train), (X_test, Y_test)

    def scale(self, training_set, testing_set, validation_set):
        (X_train, y_train), (X_test, y_test), (X_val, y_val) = training_set, testing_set, validation_set

        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(exclude=[object]).columns

        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        # Preprocess the features
        columns = numeric_features.tolist()
        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)
        X_val = pd.DataFrame(preprocessor.transform(X_val), columns=columns)

        # Preprocess the labels
        le = LabelEncoder()

        y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_test = pd.DataFrame(le.transform(y_test), columns=["label"])
        y_val = pd.DataFrame(le.transform(y_val), columns=["label"])

        # le.fit(['Benign', 'Portmap', 'DDos', 'Syn', 'TFTP', 'UDPLag'])
        # le.fit(['Benign', 'Attack'])
        # print(le.transform(['Benign', 'Attack']))
        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    def balance_dataset(self, X, y, undersampling_strategy=None, oversampling_strategy=None):
        '''
        trong tập dataset thì số lượng sample cho từng mẫu có thể ko phải lúc nào cũng tỉ lệ 50:50,
        có thể là 70:30 hoặc 90:10 => mất cân bằng dữ liệu,
        có thể làm cho việc đánh giá model trở nên, không chính xác: accuracy của model là 90% thì có thể
        90% đó đa phần là predict trên 90% dữ liệu của lớp đa số,


        Label encoder transformation:
            {
                'Benign': 0,
                'DDos': 1,
                'Portmap': 2,
                'Syn': 3,
                'TFTP': 4,
                'UDPLag': 5
            }
        '''

        # under_sampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=0)
        # X_under, y_under = under_sampler.fit_resample(X, y)

        over_sampler = SMOTE(sampling_strategy=oversampling_strategy)
        X_bal, y_bal = over_sampler.fit_resample(X, y)

        return X_bal, y_bal


if __name__ == '__main__':
    cicddos2019 = CICDdOS2019Preprocessor(DATA_DIR,
                                          training_size=0.75,
                                          testing_size=0.125,
                                          validation_size=0.125)

    cicddos2019.read_data()

    print('Size before processing: ', cicddos2019.data.shape)

    # Remove NaN, -Inf, +Inf, Duplicates
    cicddos2019.remove_duplicate_values()
    cicddos2019.remove_missing_values()
    cicddos2019.remove_infinite_values()

    # Drop constant & correlated features
    cicddos2019.remove_constant_features()
    cicddos2019.remove_correlated_features()

    # Create new label category
    cicddos2019.group_labels()

    # Split & Normalise data sets
    training_set, testing_set, validation_set = cicddos2019.train_valid_test_split()
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = cicddos2019.scale(training_set, testing_set, validation_set)

    # Balance the training set
    # 0: Attack, 1: Benign
    # oversampling_strategy = {0: 213034, 1: 1121183, 2: 132660, 3: 135876, 4: 141662, 5: 137113}
    oversampling_strategy= {0: 84400}
    X_train, y_train = cicddos2019.balance_dataset(X_train, y_train, oversampling_strategy=oversampling_strategy)

    # Visualize data
    fig, ax = plt.subplots(figsize=(15, 10))
    y_count = y_train.value_counts()
    labels = {0: 'Benign', 2: 'Portmap', 1: 'DDos', 3: 'Syn', 4: 'TFTP', 5: 'UDPLag'}
    # labels = {0: 'Attack', 1: 'Benign'}
    indexes = np.arange(len(labels))
    width = 0.4
    rect = plt.bar(indexes, [y_count[index] for index in indexes], width, color="steelblue", label="Class count")


    def add_text(rect):
        """Add text to top of each bar."""
        for r in rect:
            h = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2, h * 1.01, s=format(h, ","), fontsize=12, ha='center', va='bottom')


    add_text(rect)
    ax.set_xticks(indexes)
    ax.set_xticklabels([labels[i] for i in indexes])
    plt.xlabel('Type of attack', fontsize=16)
    plt.ylabel('# instances', fontsize=16)
    plt.legend()
    fig.savefig(os.path.join(IMG_DIR, 'classes_count_dataset.pdf'))

    # # Save the results
    print('Start saving..')
    X_train.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features.pkl'))
    X_test.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features.pkl'))
    X_val.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features.pkl'))

    y_train.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_labels.pkl'))
    y_test.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_labels.pkl'))
    y_val.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_labels.pkl'))
    print('Done saving!')

    print('\nSize after processing: ', cicddos2019.data.shape)
