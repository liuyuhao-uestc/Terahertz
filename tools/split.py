import os
import random
import argparse


def generate_indices(total_images=30000, train_size=25000, test_size=5000,
                     output_dir='./', prefix='imgHQ', suffix='.npy', zero_padding=5):
    """
    生成指定格式的索引文件

    参数:
        total_images: 总图片数量
        train_size: 训练集大小
        test_size: 测试集大小
        output_dir: 输出目录
        prefix: 文件名前缀
        suffix: 文件名后缀
        zero_padding: 数字部分零填充位数
    """
    # 验证参数
    if train_size + test_size != total_images:
        raise ValueError(f"训练集大小({train_size}) + 测试集大小({test_size}) 不等于总图片数({total_images})")

    if train_size <= 0 or test_size <= 0:
        raise ValueError("训练集和测试集大小必须大于0")

    if total_images <= 0:
        raise ValueError("总图片数必须大于0")

    # 生成所有图片的索引
    all_indices = list(range(total_images))

    # 随机打乱索引
    random.seed(789)
    random.shuffle(all_indices)

    # 分割索引
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:train_size + test_size]

    # 对索引进行排序（可选）
    train_indices.sort()
    test_indices.sort()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入训练集索引文件
    train_file = os.path.join(output_dir, 'train_files.txt')
    with open(train_file, 'w') as f:
        for idx in train_indices:
            # 生成格式化的文件名: imgHQ00000.npy
            filename = f"{prefix}{str(idx).zfill(zero_padding)}{suffix}"
            f.write(f"{filename}\n")

    # 写入测试集索引文件
    test_file = os.path.join(output_dir, 'test_files.txt')
    with open(test_file, 'w') as f:
        for idx in test_indices:
            filename = f"{prefix}{str(idx).zfill(zero_padding)}{suffix}"
            f.write(f"{filename}\n")

    print(f"数据集分割完成！")
    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    print(f"训练集文件: {train_file}")
    print(f"测试集文件: {test_file}")

    # 显示示例
    if train_indices:
        print(f"\n训练集示例（前5个）:")
        for idx in train_indices[:5]:
            print(f"  {prefix}{str(idx).zfill(zero_padding)}{suffix}")

    if test_indices:
        print(f"\n测试集示例（前5个）:")
        for idx in test_indices[:5]:
            print(f"  {prefix}{str(idx).zfill(zero_padding)}{suffix}")

    return {
        'train_file': train_file,
        'test_file': test_file,
        'train_count': len(train_indices),
        'test_count': len(test_indices)
    }


def generate_from_existing_files(image_dir, train_ratio=0.8333, output_dir='./'):
    """
    根据实际存在的.npy文件生成索引
    """
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]

    if not npy_files:
        raise ValueError(f"在目录 {image_dir} 中未找到.npy文件")

    # 按文件名排序（假设文件名类似 imgHQ00000.npy）
    npy_files.sort()

    total_files = len(npy_files)
    print(f"找到 {total_files} 个.npy文件")

    # 显示文件范围
    if npy_files:
        print(f"文件范围: {npy_files[0]} 到 {npy_files[-1]}")

    # 计算训练集和测试集大小
    train_size = int(total_files * train_ratio)
    test_size = total_files - train_size

    # 随机打乱文件列表
    random.shuffle(npy_files)

    # 分割文件
    train_files = npy_files[:train_size]
    test_files = npy_files[train_size:]

    # 对文件名进行排序（可选）
    train_files.sort()
    test_files.sort()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入训练集文件
    train_file = os.path.join(output_dir, 'train_files.txt')
    with open(train_file, 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")

    # 写入测试集文件
    test_file = os.path.join(output_dir, 'test_files.txt')
    with open(test_file, 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")

    print(f"\n数据集分割完成！")
    print(f"总文件数: {total_files}")
    print(f"训练集大小: {len(train_files)} ({len(train_files) / total_files:.2%})")
    print(f"测试集大小: {len(test_files)} ({len(test_files) / total_files:.2%})")
    print(f"训练集文件: {train_file}")
    print(f"测试集文件: {test_file}")

    return {
        'train_file': train_file,
        'test_file': test_file,
        'train_count': len(train_files),
        'test_count': len(test_files)
    }


def verify_files_exist(file_list_path, data_dir):
    """
    验证索引文件中的文件是否实际存在
    """
    if not os.path.exists(file_list_path):
        print(f"错误: 索引文件 {file_list_path} 不存在")
        return

    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    missing_files = []
    existing_files = []

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)

    print(f"验证结果 ({os.path.basename(file_list_path)}):")
    print(f"  总文件数: {len(files)}")
    print(f"  存在文件: {len(existing_files)}")
    print(f"  缺失文件: {len(missing_files)}")

    if missing_files:
        print(f"  缺失文件示例（最多显示5个）:")
        for file in missing_files[:5]:
            print(f"    {file}")

    return len(existing_files), len(missing_files)


def main():
    parser = argparse.ArgumentParser(description='分割数据集为训练集和测试集')
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'existing', 'verify'],
                        help='模式: generate(生成索引), existing(基于现有文件), verify(验证文件)')
    parser.add_argument('--total', type=int, default=30000,
                        help='总图片数量')
    parser.add_argument('--train-size', type=int, default=25000,
                        help='训练集大小')
    parser.add_argument('--test-size', type=int, default=5000,
                        help='测试集大小')
    parser.add_argument('--data-dir', type=str, default='./',
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./splits',
                        help='输出目录')
    parser.add_argument('--prefix', type=str, default='imgHQ',
                        help='文件名前缀')
    parser.add_argument('--suffix', type=str, default='.npy',
                        help='文件名后缀')
    parser.add_argument('--padding', type=int, default=5,
                        help='数字部分零填充位数')
    parser.add_argument('--train-ratio', type=float, default=25000 / 30000,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--list-file', type=str,
                        help='要验证的索引文件路径（verify模式使用）')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    if args.mode == 'generate':
        # 生成索引模式
        generate_indices(
            total_images=args.total,
            train_size=args.train_size,
            test_size=args.test_size,
            output_dir=args.output_dir,
            prefix=args.prefix,
            suffix=args.suffix,
            zero_padding=args.padding
        )

    elif args.mode == 'existing':
        # 基于现有文件模式
        generate_from_existing_files(
            image_dir=args.data_dir,
            train_ratio=args.train_ratio,
            output_dir=args.output_dir
        )

    elif args.mode == 'verify':
        # 验证模式
        if not args.list_file:
            print("请使用 --list-file 参数指定要验证的索引文件")
            return

        verify_files_exist(args.list_file, args.data_dir)


if __name__ == "__main__":
    main()