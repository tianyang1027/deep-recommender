# 数据准备示例
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreparer:
    def __init__(self, data):
        self.data = data.copy()
        self.processed_data = None
    
    def clean_data(self):
        """数据清洗"""
        print("开始数据清洗...")
        
        # 1. 处理缺失值
        print(f"处理前缺失值数量：{self.data.isnull().sum().sum()}")
        
        # 数值列用均值填充
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.data[col].isnull().sum() > 0:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        
        # 类别列用众数填充
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if self.data[col].isnull().sum() > 0:
                mode_val = self.data[col].mode()[0]
                self.data[col].fillna(mode_val, inplace=True)
        
        print(f"处理后缺失值数量：{self.data.isnull().sum().sum()}")
        
        # 2. 处理重复值
        duplicates_before = self.data.duplicated().sum()
        self.data.drop_duplicates(inplace=True)
        duplicates_after = self.data.duplicated().sum()
        print(f"删除重复值：{duplicates_before - duplicates_after} 条")
        
        # 3. 处理异常值（简单方法：使用 IQR）
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.data[col] < lower_bound) | 
                       (self.data[col] > upper_bound)).sum()
            if outliers > 0:
                # 用边界值替换异常值
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                print(f"处理 {col} 列的 {outliers} 个异常值")
        
        return self.data
    
    def feature_engineering(self):
        """特征工程"""
        print("\n开始特征工程...")
        
        # 1. 创建新特征（示例）
        if 'price' in self.data.columns and 'rating' in self.data.columns:
            # 创建性价比特征
            self.data['price_per_rating'] = self.data['price'] / self.data['rating']
            print("创建新特征：price_per_rating")
        
        # 2. 特征选择（简单示例：移除低方差特征）
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        low_variance_features = []
        
        for col in numeric_columns:
            if self.data[col].var() < 0.01:  # 方差阈值
                low_variance_features.append(col)
        
        if low_variance_features:
            self.data.drop(columns=low_variance_features, inplace=True)
            print(f"移除低方差特征：{low_variance_features}")
        
        return self.data
    
    def transform_data(self):
        """数据转换"""
        print("\n开始数据转换...")
        
        # 1. 编码类别变量
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le
            print(f"编码类别变量：{col}")
        
        # 2. 标准化数值变量
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        
        if len(numeric_columns) > 0:
            self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
            print(f"标准化数值变量：{list(numeric_columns)}")
        
        return self.data, label_encoders, scaler
    
    def split_data(self, target_column, test_size=0.2, val_size=0.2):
        """数据划分"""
        print(f"\n开始数据划分（测试集比例：{test_size}，验证集比例：{val_size}）...")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 再从剩余数据中分离出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        print(f"训练集大小：{X_train.shape[0]}")
        print(f"验证集大小：{X_val.shape[0]}")
        print(f"测试集大小：{X_test.shape[0]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def prepare_pipeline(self, target_column):
        """完整的数据准备流水线"""
        print("=" * 50)
        print("数据准备流水线")
        print("=" * 50)
        
        # 1. 数据清洗
        self.clean_data()
        
        # 2. 特征工程
        self.feature_engineering()
        
        # 3. 数据转换
        processed_data, encoders, scaler = self.transform_data()
        
        # 4. 数据划分
        splits = self.split_data(target_column)
        
        self.processed_data = processed_data
        return splits, encoders, scaler

# 创建示例数据并演示数据准备
np.random.seed(42)
sample_data = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'gender': np.random.choice(['男', '女'], 1000),
    'city': np.random.choice(['北京', '上海', '广州'], 1000),
    'target': np.random.choice([0, 1], 1000)
})

# 添加一些缺失值和异常值
sample_data.loc[np.random.choice(1000, 50), 'income'] = np.nan
sample_data.loc[np.random.choice(1000, 20), 'age'] = np.random.randint(100, 150)

preparer = DataPreparer(sample_data)
splits, encoders, scaler = preparer.prepare_pipeline('target')