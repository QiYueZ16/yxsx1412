# print_tree.py
import os

def print_tree(startpath, prefix="", depth=0, max_depth=4, exclude_dirs=None, file=None):
    """
    递归打印目录树，仅展开到 max_depth 级（根目录 depth=0），
    并跳过所有在 exclude_dirs 中指定的文件夹（不展开其子内容）。
    """
    if depth > max_depth:
        return

    if exclude_dirs is None:
        exclude_dirs = {"results"}   # 默认不展开 results 目录

    try:
        files = [f for f in os.listdir(startpath) if f != "__pycache__" and not f.endswith(".pyc")]
        files.sort()
    except PermissionError:
        return

    for i, f in enumerate(files):
        path = os.path.join(startpath, f)
        connector = "└── " if i == len(files) - 1 else "├── "
        line = prefix + connector + f
        print(line)
        if file:
            file.write(line + "\n")

        # 如果是目录且需要展开
        if os.path.isdir(path):
            extension = "    " if i == len(files) - 1 else "│   "
            # 仅当当前深度 < max_depth 且 该目录名不在排除列表中时，才递归进入
            if depth < max_depth and f not in exclude_dirs:
                print_tree(path, prefix + extension, depth + 1, max_depth, exclude_dirs, file)

if __name__ == "__main__":
    root_dir = "."
    output_file = "tree.txt"

    abs_root = os.path.abspath(root_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(abs_root + "/\n")
        print(abs_root + "/")
        # 展开到第 4 级，且排除 results 目录
        print_tree(root_dir, max_depth=4, exclude_dirs={"results"}, file=f)

    print(f"\n✅ 文件树已保存到 {output_file}")