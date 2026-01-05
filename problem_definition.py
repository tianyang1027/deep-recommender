# 问题定义示例：电商推荐系统
class ProblemDefinition:
    def __init__(self):
        # 业务问题
        self.business_problem = "用户购买转化率低，需要提高推荐精准度"
        
        # 技术问题
        self.technical_problem = "基于用户行为预测用户可能购买的商品"
        
        # 问题类型
        self.problem_type = "推荐系统（分类+排序）"
        
        # 成功标准
        self.success_criteria = {
            "点击率提升": "15%",
            "转化率提升": "10%",
            "推荐准确率": "80%"
        }
        
        # 约束条件
        self.constraints = {
            "响应时间": "< 100ms",
            "数据隐私": "符合 GDPR 要求",
            "计算资源": "现有服务器配置"
        }
    
    def define_features_and_labels(self):
        """定义特征和标签"""
        features = {
            "用户特征": ["年龄", "性别", "购买历史", "浏览行为"],
            "商品特征": ["类别", "价格", "评分", "库存"],
            "上下文特征": ["时间", "设备", "地理位置"]
        }
        
        labels = {
            "主要标签": "是否点击",
            "次要标签": "是否购买",
            "辅助标签": "停留时间"
        }
        
        return features, labels
    
    def print_definition(self):
        """打印问题定义"""
        print("=" * 50)
        print("机器学习问题定义")
        print("=" * 50)
        print(f"业务问题：{self.business_problem}")
        print(f"技术问题：{self.technical_problem}")
        print(f"问题类型：{self.problem_type}")
        print("\n成功标准：")
        for metric, target in self.success_criteria.items():
            print(f"  {metric}：{target}")
        print("\n约束条件：")
        for constraint, limit in self.constraints.items():
            print(f"  {constraint}：{limit}")
        
        features, labels = self.define_features_and_labels()
        print("\n特征定义：")
        for category, items in features.items():
            print(f"  {category}：{', '.join(items)}")
        print("\n标签定义：")
        for label_type, label_name in labels.items():
            print(f"  {label_type}：{label_name}")

# 使用示例
problem = ProblemDefinition()
problem.print_definition()