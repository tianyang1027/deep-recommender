# 模型评估示例
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """评估分类模型"""
        results = {}
        
        # 基本指标
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n{model_name} 分类评估结果：")
        print(f"准确率：{results['accuracy']:.4f}")
        print(f"精确率：{results['precision']:.4f}")
        print(f"召回率：{results['recall']:.4f}")
        print(f"F1 分数：{results['f1']:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩阵：")
        print(cm)
        
        # ROC 曲线（如果有概率预测）
        if y_prob is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            results['roc_auc'] = roc_auc
            
            # 绘制 ROC 曲线
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title(f'{model_name} ROC 曲线')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        
        self.evaluation_results[model_name] = results
        return results
    
    def evaluate_regression(self, y_true, y_pred, model_name="Model"):
        """评估回归模型"""
        results = {}
        
        # 基本指标
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² 分数
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        results['mse'] = mse
        results['rmse'] = rmse
        results['mae'] = mae
        results['r2'] = r2
        
        print(f"\n{model_name} 回归评估结果：")
        print(f"均方误差 (MSE)：{mse:.4f}")
        print(f"均方根误差 (RMSE)：{rmse:.4f}")
        print(f"平均绝对误差 (MAE)：{mae:.4f}")
        print(f"R² 分数：{r2:.4f}")
        
        # 绘制预测 vs 真实值
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} 预测 vs 真实值')
        plt.grid(True)
        plt.show()
        
        self.evaluation_results[model_name] = results
        return results
    
    def compare_models(self):
        """比较所有评估过的模型"""
        if not self.evaluation_results:
            print("没有可比较的模型评估结果")
            return
        
        print("\n模型比较：")
        print("-" * 50)
        
        # 创建比较表格
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            row = [model_name]
            for metric, value in results.items():
                row.append(f"{value:.4f}")
            comparison_data.append(row)
        
        # 打印表格
        headers = ["模型名称"] + list(self.evaluation_results.values())[0].keys()
        print("\t".join(headers))
        for row in comparison_data:
            print("\t".join(row))

# 使用示例
evaluator = ModelEvaluator()

# 评估分类模型
y_pred_class = best_model.predict(X_test)
y_prob_class = best_model.predict_proba(X_test)
evaluator.evaluate_classification(y_test, y_pred_class, y_prob_class, "最佳分类模型")

# 比较所有模型
evaluator.compare_models()