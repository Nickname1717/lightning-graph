def test_writing_to_directory(directory_path):
    try:
        # 尝试在指定目录中创建一个文本文件并写入数据
        with open(directory_path + "/test_file.txt", "w") as file:
            file.write("This is a test.")
        print("写入成功！")
    except Exception as e:
        print(f"写入失败: {e}")

# 请替换成你想要测试的目录路径
directory_to_test = "D:/zjclearning/pythonproject/lightning-graph/ckpt"

test_writing_to_directory(directory_to_test)
