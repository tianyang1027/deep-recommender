# 模型训练示例
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
    
    def register_model(self, name, model, problem_type):
        """注册模型"""
        self.models[name] = {
            'model': model,
            'problem_type': problem_type
        }
        print(f"注册模型：{name}（{problem_type}）")
    
    def train_single_model(self, name, X_train, y_train):
        """训练单个模型"""
        if name not in self.models:
            raise ValueError(f"模型 {name} 未注册")
        
        model_info = self.models[name]
        model = model_info['model']
        
        print(f"\n训练模型：{name}")
        model.fit(X_train, y_train)
        
        self.trained_models[name] = model
        print(f"模型 {name} 训练完成")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """训练所有注册的模型"""
        print("\n开始训练所有模型...")
        
        for name in self.models.keys():
            try:
                self.train_single_model(name, X_train, y_train)
            except Exception as e:
                print(f"训练模型 {name} 时出错：{e}")
        
        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """评估所有训练好的模型"""
        print("\n模型评估结果：")
        print("-" * 50)
        
        results = {}
        
        for name, model in self.trained_models.items():
            problem_type = self.models[name]['problem_type']
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 根据问题类型选择评估指标
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {'accuracy': accuracy}
                print(f"{name}: 准确率 = {accuracy:.4f}")
                
                # 详细报告
                print(classification_report(y_test, y_pred))
                
            elif problem_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                results[name] = {'mse': mse, 'rmse': rmse}
                print(f"{name}: MSE = {mse:.4f}, RMSE = {rmse:.4f}")
            
            print("-" * 50)
        
        return results
    
    def get_best_model(self, results, metric='accuracy'):
        """获取最佳模型"""
        if not results:
            return None
        
        best_model_name = max(results.keys(), key=lambda x: results[x].get(metric, 0))
        best_score = results[best_model_name][metric]
        
        print(f"\n最佳模型：{best_model_name}（{metric} = {best_score:.4f}）")
        
        return best_model_name, self.trained_models[best_model_name]

# 使用示例
trainer = ModelTrainer()

# 注册不同类型的模型
trainer.register_model('逻辑回归', LogisticRegression(random_state=42), 'classification')
trainer.register_model('随机森林', RandomForestClassifier(n_estimators=100, random_state=42), 'classification')
trainer.register_model('支持向量机', SVC(random_state=42), 'classification')

# 创建训练数据
X_train = splits['X_train']
y_train = splits['y_train']
X_test = splits['X_test']
y_test = splits['y_test']

# 训练所有模型
trained_models = trainer.train_all_models(X_train, y_train)

# 评估模型
results = trainer.evaluate_models(X_test, y_test)

# 获取最佳模型
best_name, best_model = trainer.get_best_model(results)