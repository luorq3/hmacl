import argparse

# 创建一个父级 ArgumentParser 对象，其中包含两个参数
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--verbose', '-v', action='store_true', help='打印详细信息')
parent_parser.add_argument('--debug', '-d', action='store_true', help='打印调试信息')

# 创建一个子级 ArgumentParser 对象，并将父级 ArgumentParser 对象作为参数传递
parser = argparse.ArgumentParser(description='我的程序')
parser.add_argument('--input', '-i', type=str, help='输入文件')
parser.add_argument('--output', '-o', type=str, help='输出文件')
parser.add_argument('--overwrite', action='store_true', help='覆盖输出文件')
parser.add_argument('--format', type=str, default='txt', help='输出格式')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.parents = [parent_parser]

# 解析命令行参数
args = parser.parse_args()

# 现在 args 中将包含父级 ArgumentParser 对象中定义的参数，以及子级 ArgumentParser 对象中定义的参数。
# 您可以像这样访问这些参数：
# if args.verbose:
#     print('打印详细信息')
# if args.debug:
#     print('打印调试信息')
# if args.input:
#     print('输入文件:', args.input)
# if args.output:
#     print('输出文件:', args.output)
# if args.overwrite:
#     print('覆盖输出文件')
# print('输出格式:', args.format)
print(args)
print(parent_parser.parse_args())
