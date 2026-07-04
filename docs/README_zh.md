英文版本请见：[`README.md`](./README.md)

# MagiCompiler 文档指南

本指南说明如何构建、预览和贡献 MagiCompiler 文档。无需任何 Sphinx 经验即可上手。

## 前置条件

- Python 3.10+
- `pip`（Python 自带）
- 终端（bash、zsh、PowerShell 等均可）
- 文本编辑器（VS Code、Vim 等均可）

## 快速上手（5 分钟）

```bash
# 1. 进入 docs 目录
cd docs

# 2. 安装依赖（仅首次需要）
pip install -r requirements.txt

# 3. 构建文档
make html

# 4. 在浏览器中打开
open build/html/index.html        # macOS
xdg-open build/html/index.html    # Linux
start build/html/index.html       # Windows (Git Bash)
```

完成！你应该能在浏览器中看到文档了。

> **提示：** 如果看到旧内容，运行 `make clean && make html` 强制全量重建。

---

## 目录结构

```
docs/
├── source/                  # 所有文档源文件都在这里
│   ├── index.md             # 首页
│   ├── conf.py              # Sphinx 配置文件（一般不需要改）
│   ├── _static/             # 自定义 CSS、文档引用的图片
│   ├── _templates/          # HTML 模板（如语言切换器）
│   ├── user_guide/          # 面向用户的指南
│   │   ├── toc.md           # 用户指南章节的目录
│   │   ├── install.md       # 安装指南
│   │   └── quickstart.md    # 快速开始
│   └── blog/                # 技术博客文章
│       ├── toc.md           # 博客章节的目录
│       └── refs/            # BibTeX 引用文件（每篇博客一个）
├── locale/                  # 中文翻译文件（.po）
│   └── zh_CN/LC_MESSAGES/
├── build/                   # 生成的输出（已被 git 忽略）
├── requirements.txt         # Python 依赖
├── Makefile                 # 构建命令（Linux/macOS）
└── make.bat                 # 构建命令（Windows）
```

---

## 构建命令

| 命令 | 说明 |
|------|------|
| `make html` | 构建英文文档（默认） |
| `make html-en` | 构建英文文档到 `build/html`（站点根目录） |
| `make html-zh` | 构建中文文档到 `build/html/zh_CN` |
| `make html-multilang` | 同时构建英文和中文文档 |
| `make update-po` | 重新生成 `.po` 翻译模板 |
| `make clean` | 删除 `build/` 目录 |

---

## 如何编写文档

所有文档都使用 **Markdown** 编写，采用 [MyST 语法](https://myst-parser.readthedocs.io/en/latest/)——它在标准 Markdown 基础上扩展了 Sphinx 特有的功能。

### 新增一篇用户指南

1. 在 `source/user_guide/` 下创建一个新的 `.md` 文件。
2. 在 `source/user_guide/toc.md` 中注册其文件名（不含 `.md`）。
3. 用 `make clean && make html` 构建并预览。

### 新增一篇博客文章

1. 创建带 YAML frontmatter 头部（`blogpost: true`、`date`、`author` 等）的
   `source/blog/my_topic.md`。
2. 如果包含引用文献，创建 `source/blog/refs/my_topic.bib`，并在
   `source/conf.py` 的 `blog_titles` 列表中添加该标题。
3. 在 `source/blog/toc.md` 中注册。
4. 构建并预览。

---

## 中英文双语文档

所有文档以**英文为主源**，中文翻译通过 `.po` 文件管理，遵循标准的 Sphinx 国际化（i18n）工作流。

```
source/*.md（英文源文件，"唯一真相源"）
     │
     ▼  make update-po
locale/zh_CN/LC_MESSAGES/*.po（中文翻译）
     │
     ▼  make html-multilang
build/html/        （英文站点，位于站点根目录）
build/html/zh_CN/  （中文站点，带语言切换器）
```

1. 照常编辑 `source/` 下的英文 `.md` 源文件。
2. 运行 `make update-po` 提取/刷新翻译字符串。
3. 在 `locale/zh_CN/LC_MESSAGES/*.po` 中翻译 `msgstr` 条目
   （不要修改 `msgid`）。
4. 运行 `make html-multilang` 同时构建两种语言。
