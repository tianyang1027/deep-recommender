# 数据收集示例：模拟多种数据源
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.collected_data = {}
    
    def collect_user_data(self, n_users=1000):
        """收集用户数据"""
        np.random.seed(42)
        
        user_data = {
            'user_id': range(1, n_users + 1),
            'age': np.random.randint(18, 65, n_users),
            'gender': np.random.choice(['男', '女'], n_users),
            'city': np.random.choice(['北京', '上海', '广州', '深圳'], n_users),
            'registration_date': [
                datetime.now() - timedelta(days=np.random.randint(1, 365))
                for _ in range(n_users)
            ]
        }
        
        self.collected_data['users'] = pd.DataFrame(user_data)
        print(f"收集了 {len(user_data['user_id'])} 条用户数据")
        return self.collected_data['users']
    
    def collect_behavior_data(self, n_behaviors=5000):
        """收集用户行为数据"""
        np.random.seed(42)
        
        user_ids = np.random.choice(range(1, 1001), n_behaviors)
        product_ids = np.random.choice(range(1, 501), n_behaviors)
        
        behavior_data = {
            'behavior_id': range(1, n_behaviors + 1),
            'user_id': user_ids,
            'product_id': product_ids,
            'behavior_type': np.random.choice(
                ['浏览', '点击', '加购物车', '购买'], n_behaviors, 
                p=[0.4, 0.3, 0.2, 0.1]
            ),
            'timestamp': [
                datetime.now() - timedelta(minutes=np.random.randint(1, 10080))
                for _ in range(n_behaviors)
            ],
            'duration': np.random.exponential(30, n_behaviors)  # 停留时间（秒）
        }
        
        self.collected_data['behaviors'] = pd.DataFrame(behavior_data)
        print(f"收集了 {len(behavior_data['behavior_id'])} 条行为数据")
        return self.collected_data['behaviors']
    
    def collect_product_data(self, n_products=500):
        """收集商品数据"""
        np.random.seed(42)
        
        categories = ['电子产品', '服装', '食品', '家居', '图书']
        product_data = {
            'product_id': range(1, n_products + 1),
            'category': np.random.choice(categories, n_products),
            'price': np.random.uniform(10, 1000, n_products),
            'rating': np.random.uniform(3.0, 5.0, n_products),
            'stock': np.random.randint(0, 1000, n_products)
        }
        
        self.collected_data['products'] = pd.DataFrame(product_data)
        print(f"收集了 {len(product_data['product_id'])} 条商品数据")
        return self.collected_data['products']
    
    def get_data_summary(self):
        """获取数据摘要"""
        print("\n数据收集摘要：")
        for name, df in self.collected_data.items():
            print(f"\n{name} 数据集：")
            print(f"  形状：{df.shape}")
            print(f"  列名：{list(df.columns)}")
            print(f"  缺失值：{df.isnull().sum().sum()}")
            print(f"  示例数据：")
            print(df.head(2))

# 使用示例
collector = DataCollector()
collector.collect_user_data()
collector.collect_behavior_data()
collector.collect_product_data()
collector.get_data_summary()