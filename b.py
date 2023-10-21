import os

def replace_string_in_filenames(directory, old_string, new_string):
    for filename in os.listdir(directory):
        if old_string in filename:
            new_filename = filename.replace(old_string, new_string)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"{filename} 已更名为 {new_filename}")

# 指定目录路径
directory_path = "./Network/VIT_Model_cifar100"

# 指定要替换的字符串和新字符串
old_string = "Orain"
new_string = "orain"

# 调用函数进行替换
replace_string_in_filenames(directory_path, old_string, new_string)