import os
import glob
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 80)


def shorten(x, n1=30, n2=30):
    s = str(x)
    if len(s) <= n1 + n2 + 10:
        return s
    return s[:n1] + " ... " + s[-n2:]


def main():
    # 强行指定你截图里的路径
    base_dir = r"C:\Users\TYTOO\Desktop\lgbtq-hate"

    print("1. 检查根目录是否存在:", os.path.exists(base_dir))
    if not os.path.exists(base_dir):
        print("❌ 根目录都不存在，说明你的桌面可能被 OneDrive 重定向了。")
        print(
            "👉 请在 PyCharm 左侧列表里，右键 lgbtq-hate 文件夹 -> Copy Path/Reference -> Absolute Path，把那个路径覆盖到代码里的 base_dir。")
        return

    print("2. 根目录下的实际内容:", os.listdir(base_dir))

    print("\n3. 开始深度搜索所有的 .tsv 文件...")
    # 使用 glob 递归搜索所有 .tsv 文件
    tsv_files = glob.glob(os.path.join(base_dir, "**", "*.tsv"), recursive=True)

    if not tsv_files:
        print("❌ 依然没有找到任何 .tsv 文件！请确认你的数据到底放在哪了。")
        return

    print(f"✅ 找到了！共 {len(tsv_files)} 个文件：")
    for f in tsv_files:
        print("  -", f)

    print("\n4. 开始解析数据...")
    for path in tsv_files:
        print("\n" + "=" * 80)
        print("FILE:", os.path.basename(path))
        print("=" * 80)

        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

        print("shape:", df.shape)
        print("columns:", df.columns.tolist())

        print("\n前两行 (文本已截断):")
        head = df.head(2).copy()
        for col in head.columns:
            low = col.lower()
            if any(k in low for k in ["comment", "text", "description", "title"]):
                head[col] = head[col].apply(shorten)
        print(head.to_string(index=False))

        print("\n各列的唯一值分布 (仅展示类别较少的列):")
        for col in df.columns:
            nun = df[col].nunique(dropna=True)
            if nun <= 20:
                print(f"--- {col} (nunique={nun}) ---")
                print(df[col].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
