import pandas as pd

def cut_data():
    # 定义文件路径
    file_path = "datasets/train_first_half.parquet"
    
    # 读取数据
    data = pd.read_parquet(file_path)
    
    # 在 problem 列的每个问题前加上 "Return your final response within \\boxed{}."
    # data['problem'] = data['problem'].apply(lambda x: "Return your final response within \\boxed{}. " + x)
    
    # 打印修改后的第一个问题
    print(data['problem'][0])
    
    # 将数据集拆分成两半
    half_size = len(data) // 2
    data_first_half = data.iloc[:half_size]
    data_second_half = data.iloc[half_size:]
    
    # 打印每部分的长度
    print(f"First half length: {len(data_first_half)}")
    print(f"Second half length: {len(data_second_half)}")
    
    # 保存到新的 Parquet 文件
    data_first_half.to_parquet("datasets/train_1_in_4.parquet", index=False)
    data_second_half.to_parquet("datasets/train_2_in_4.parquet", index=False)

# 调用函数
# cut_data()

def formatted_data():
        # 定义文件路径
    file_path = "datasets/train_first_half.parquet"
    
    # 读取数据
    data = pd.read_parquet(file_path)
    
    # 在 problem 列的每个问题前加上 "Return your final response within \\boxed{}."
    data['problem'] = data['problem'].apply(lambda x: "Return your final response within \\boxed{}. " + x)
    
    # 打印修改后的第一个问题
    print(data['problem'][0])
    
    # 保存到新的 Parquet 文件
    target_path = file_path.replace(".parquet", "_formatted.parquet")
    data.to_parquet(target_path, index=False)


def visualize_data():
        # 定义文件路径
    file_path = "/home/share/wenhao/datasets/hiyouga_math12k/train-00000-of-00001.parquet"
    
    # 读取数据
    data = pd.read_parquet(file_path)

    print(data.head())

    
if __name__ == "__main__":
    # formatted_data()
    # visualize_data()
    cut_data()