# 安装说明

## 发布内容

- `app.py`：单进程 Web 服务，内置 API 与前端静态页面
- `web/`：已构建前端
- `legal_pii_redactor/`：核心脱敏库源码
- `wheels/`：Python wheel 安装包
- `samples/`：示例文件

## 环境要求

- Windows 10/11
- Python 3.10 或更高
- 网络可访问 PyPI 镜像，用于安装 Flask / pdfplumber 等依赖

## 最快安装

1. 双击 `install.bat`
2. 等待依赖安装完成
3. 双击 `start_web.bat`
4. 浏览器打开 `http://127.0.0.1:5000`

## 命令行安装

```powershell
cd "legal-pii-redactor-complete-v0.2.0"
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## CLI 库安装

如果只想安装脱敏库命令行：

```powershell
cd "legal-pii-redactor-complete-v0.2.0"
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install wheels\legal_pii_redactor-0.2.0-py3-none-any.whl
legal-redact --text "被告人张三，身份证号110101199001011234"
```

## 默认模式

- `start_web.bat` 默认使用纯正则模式
- 纯正则模式依赖最少，当前发布包已完成验证
- `start_web_ner.bat` 会尝试启用 HanLP NER，但你需要自行补装 `hanlp`、`torch`、`transformers`

## 可选 NER 安装

```powershell
.venv\Scripts\Activate.ps1
pip install hanlp torch "transformers<5"
$env:LEGAL_PII_USE_NER = "1"
python app.py
```

## 常见问题

- 端口占用：先关闭占用 `5000` 端口的程序，再重启 `start_web.bat`
- PDF 解析为空：扫描版 PDF 需要先做 OCR，本包当前只处理可提取文本的 PDF
- NER 启动失败：先用默认纯正则模式；当前机器环境里 PyTorch DLL 初始化存在风险
