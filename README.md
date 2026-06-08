#English AI Review

一个基于 AI API 的英语复习资料生成工具。项目由两个脚本整理而来：

- `generate_writing_report.py`：生成考研英语（二）作文模板报告。
- `generate_2023_review.py`：生成 2023 考研英语（二）客观题全题复盘手册。

> 注意：本仓库不包含 API Key，不包含官方真题 PDF。请自行准备合法来源的 PDF，并在本地配置环境变量。

## 功能

- 生成 Markdown 复习手册
- 支持检查点续跑，避免中途失败后重头开始
- 支持输出目录和模型名通过环境变量配置
- 支持 GCS PDF URI，用于让 Gemini 读取 PDF
- 自动估算调用成本，仅供参考

## 安装

```bash
pip install -r requirements.txt
```

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```env
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-3.1-pro-preview
OUTPUT_DIR=outputs
CHECKPOINT_DIR=checkpoints
```

如果运行 2023 客观题复盘脚本，还需要配置 PDF 来源。推荐使用 GCS URI：

```env
GCS_URI=gs://your-bucket/23English.pdf
```

也可以分别配置：

```env
GCS_BUCKET_NAME=your-bucket
PDF_OBJECT_NAME=23English.pdf
```

## 运行

### 生成作文模板报告

```bash
python scripts/generate_writing_report.py
```

输出文件默认在：

```text
outputs/考研英语二作文_高级模板报告_V3.md
```

### 生成 2023 客观题复盘手册

```bash
python scripts/generate_2023_review.py
```

输出文件默认在：

```text
outputs/2023英语_全题复习手册_v8.2.md
```

## 数据修改

`generate_2023_review.py` 里保留了学生作答序列和标准答案序列。你可以按自己的答题情况修改：

- `CLOZE_USER`
- `READ_USER`
- `PARTB_USER`

如果你要换年份，需要同步修改题号、文章主题、答案序列和 prompt。



## License

MIT
