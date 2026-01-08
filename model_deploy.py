# 模型部署示例
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score
    


class ModelDeployer:
    def __init__(self):
        self.deployed_models = {}
        self.deployment_logs = []
    
    def save_model(self, model, model_name, filepath=None):
        """保存模型"""
        if filepath is None:
            filepath = f"{model_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"模型 {model_name} 已保存到 {filepath}")
        
        # 记录部署日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'save_model',
            'model_name': model_name,
            'filepath': filepath
        }
        self.deployment_logs.append(log_entry)
        
        return filepath
    
    def load_model(self, model_name, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.deployed_models[model_name] = model
        print(f"模型 {model_name} 已从 {filepath} 加载")
        
        # 记录部署日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'load_model',
            'model_name': model_name,
            'filepath': filepath
        }
        self.deployment_logs.append(log_entry)
        
        return model
    
    def create_prediction_service(self, model_name, encoders=None, scaler=None):
        """创建预测服务"""
        if model_name not in self.deployed_models:
            raise ValueError(f"模型 {model_name} 未部署")
        
        model = self.deployed_models[model_name]
        
        def predict_service(input_data):
            """预测服务函数"""
            try:
                # 数据预处理
                if encoders:
                    for col, encoder in encoders.items():
                        if col in input_data.columns:
                            input_data[col] = encoder.transform(input_data[col])
                
                if scaler:
                    numeric_cols = input_data.select_dtypes(include=['number']).columns
                    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
                
                # 预测
                prediction = model.predict(input_data)
                
                # 如果是分类模型，也返回概率
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)
                    return {
                        'prediction': prediction.tolist(),
                        'probability': probability.tolist(),
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'prediction': prediction.tolist(),
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    }
            
            except Exception as e:
                return {
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }
        
        # 记录服务创建日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'create_service',
            'model_name': model_name
        }
        self.deployment_logs.append(log_entry)
        
        return predict_service
    
    def monitor_model(self, model_name, input_data, true_labels=None):
        """监控模型性能"""
        if model_name not in self.deployed_models:
            raise ValueError(f"模型 {model_name} 未部署")
        
        predict_service = self.create_prediction_service(model_name)
        
        # 获取预测结果
        result = predict_service(input_data)
        
        # 监控信息
        monitoring_info = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'input_shape': input_data.shape,
            'prediction_count': len(result.get('prediction', [])),
            'status': result.get('status', 'unknown')
        }
        
        # 如果有真实标签，计算性能指标
        if true_labels is not None and 'prediction' in result:
            predictions = result['prediction']
            if len(predictions) == len(true_labels):
                accuracy = accuracy_score(true_labels, predictions)
                monitoring_info['accuracy'] = accuracy
        
        print("模型监控信息：")
        for key, value in monitoring_info.items():
            print(f"  {key}: {value}")
        
        return monitoring_info
    
    def get_deployment_logs(self):
        """获取部署日志"""
        return self.deployment_logs

# 使用示例
deployer = ModelDeployer()

# 保存最佳模型
model_path = deployer.save_model(best_model, "best_classification_model")

# 加载模型
deployer.load_model("best_classification_model", model_path)

# 创建预测服务
prediction_service = deployer.create_prediction_service(
    "best_classification_model", encoders, scaler
)

# 使用预测服务
test_input = X_test.head(5)
prediction_result = prediction_service(test_input)
print("\n预测结果：")
print(json.dumps(prediction_result, indent=2, ensure_ascii=False))

# 监控模型
deployer.monitor_model("best_classification_model", test_input, y_test.head(5).values)