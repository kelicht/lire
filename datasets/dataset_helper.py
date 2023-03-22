import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = './'

DATASETS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 's', 'w']

DATASET_NAMES = {
    'a': 'adult', 
    'b': 'bank',
    'c': 'compas',
    'd': 'diabetes',
    'e': 'employeeattrition',
    'f': 'fico',
    'g': 'german',
    'h': 'heart',
    'i': 'ionosphere',
    's': 'student',
    'w': 'winequality',
}

DATASET_FULLNAMES = {
    'a': 'Adult', 
    'b': 'Bank',
    'c': 'COMPAS',
    'd': 'Diabetes',
    'e': 'Attrition',
    'f': 'FICO',
    'g': 'German',
    'h': 'Heart',
    'i': 'Ionosphere',
    's': 'Student',
    'w': 'WineQuality',
}

TARGET_NAMES = {
    'a': 'Income', 
    'b': 'subscribe',
    'c': 'RecidivateWithinTwoYears',
    'd': 'Outcome',
    'e': 'Attrition',
    'f': 'RiskPerformance',
    'g': 'BadCustomer',
    'h': 'disease',
    'i': 'Return',
    's': 'Grade',
    'w': 'Quality',
}

FEATURE_TYPES = {
    'a': ['I']*6 + ['B']*102,
    'b': ['I']*6 + ['B']*29,
    'c': ['I']*5 + ['B']*9,
    'd': ['I'] + ['C']*6 + ['I'],
    'e': ['I']*5 + ['B'] + ['I']*5 + ['B','I','B'] + ['I']*9 + ['B']*21,
    'f': ['I']*23,
    'g': ['I']*6 + ['B']*34,
    'h': ['I'] + ['B'] + ['I'] + ['C']*2 + ['B'] + ['I']*3 + ['C'] + ['I']*2 + ['B']*3,
    'i': ['B']*2 + ['C']*32,
    's': ['I']*6 + ['B']*8 + ['I']*7 + ['B']*27,
    'w': ['C']*5 + ['I']*2 + ['C']*4 + ['B'],
}

FEATURE_CONSTRAINTS = {
    'a': ['I'] + ['']*57 + ['F']*50,
    'b': ['F']*2 + ['', 'I', '', 'I'] + ['F']*22 + ['']*7,
    'c': ['I'] + ['']*4 + ['F']*6 + ['']*2 + ['F'],
    'd': ['I'] + ['']*6 + ['I'],
    'e': ['F', '', 'F', 'F', '', 'F'] + ['']*4 + ['F'] + ['']*8 + ['I']*4 + ['']*3 + ['F']*6 + ['']*9 + ['F']*3,
    'f': ['']*23,
    'g': ['F'] + ['']*36 + ['F']*3,
    'h': ['']*15, # temp
    'i': ['']*34, # temp
    's': ['F']*4 + [''] + ['F'] + ['']*14 + ['F']*28,
    'w': ['']*11 + ['F'],
}

FEATURE_CATEGORIES = {
    'a': [list(range(6,15)), list(range(15,31)), list(range(31,38)), list(range(38,53)), list(range(53,59))],
    'b': [list(range(9, 21)), list(range(21, 24)), list(range(24, 28)), list(range(28, 31)), list(range(31, 35))],
    'c': [[5,6,7,8,9,10],[11,12]],
    'd': [],
    'e': [list(range(23, 26)), list(range(26, 32)), list(range(32, 41)), list(range(41, 44))],
    'f': [],
    'g': [[18,19,20,21,22,23,24,25,26,27],[28,29,30],[31,32,33],[34,35,36]],
    'h': [[12,13,14]], # temp
    'i': [], # temp
    's': [list(range(21,23)), list(range(23,25)), list(range(25,27)), list(range(27,29)), list(range(29,31)), list(range(31,36)), list(range(36,41)), list(range(41,45)), list(range(45,48))],
    'w': [],
}

CLASS_NAMES = {
    'a': ['Income: >50K', 'Income: <=50K'], 
    'b': ['Subscribe: Yes', 'Subscribe: No'],
    'c': ['Not Recidivate', 'Recidivate'],
    'd': ['Outcome: Good', 'Outcome: Bad'],
    'e': ['Attrition: No', 'Attrition: Yes'],
    'f': ['RiskPerformance: Good', 'RiskPerformance: Bad'],
    'g': ['Low risk of Default', 'High risk of Default'],
    'h': ['Heart disease: No', 'Heart disease: Yes'],
    'i': ['Return: Good', 'Return: Bad'],
    's': ['Grade: >10', 'Grade: <=10'],
    'w': ['Quality: >5', 'Quality: <=5'],
}



class Dataset():
    def __init__(self, dataset='g'):
        self.df = pd.read_csv(CURRENT_DIR+'datasets/{}.csv'.format(DATASET_NAMES[dataset]))
        self.y = self.df[TARGET_NAMES[dataset]].values
        self.X = self.df.drop([TARGET_NAMES[dataset]], axis=1).values
        self.dataset_name = DATASET_NAMES[dataset]
        self.dataset_fullname = DATASET_FULLNAMES[dataset]
        self.target_name = TARGET_NAMES[dataset]
        self.feature_names = list(self.df.drop([TARGET_NAMES[dataset]], axis=1).columns)
        self.feature_types = FEATURE_TYPES[dataset]
        self.feature_constraints = FEATURE_CONSTRAINTS[dataset]
        self.feature_categories = FEATURE_CATEGORIES[dataset]
        self.class_names = CLASS_NAMES[dataset]

    def get_dataset(self, split=False, test_size=0.25):
        if split:
            X_tr, X_ts, y_tr, y_ts = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)
            return X_tr, X_ts, y_tr, y_ts
        else:
            return self.X, self.y

