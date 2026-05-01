# PAN 生成式剽窃检测项目说明书（零基础版）

这份说明书面向第一次参加 PAN/TIRA 测评的同学。你只需要按步骤做，就能理解任务、跑通 baseline、用验证集测分、调参数、打 Docker 镜像，并把系统提交到网站。

## 1. 这个测评任务到底要做什么？

PAN 生成式剽窃检测是一个“文本片段对齐”任务。网站给你很多文档对：一篇是 suspicious document（可疑文档），另一篇是 source document（来源文档）。你的系统要判断 suspicious 文档中哪些连续片段来自 source 文档，并输出这些片段在两篇文档里的字符位置。

输入目录通常长这样：

```text
/input/
├── pairs
├── susp/
│   ├── suspicious-document001.txt
│   └── suspicious-document002.txt
└── src/
    ├── source-document001.txt
    └── source-document015.txt
```

`pairs` 文件每一行是一对需要比较的文档：

```text
suspicious-document001.txt source-document001.txt
suspicious-document002.txt source-document015.txt
```

你的程序要对每一行 pair 生成一个结果文件，比如：

```text
/output/suspicious-document001-source-document001.xml
```

XML 里每个 `feature` 表示一个检测到的剽窃片段：

```xml
<document reference="suspicious-document001.txt">
  <feature
    name="detected-plagiarism"
    this_offset="123"
    this_length="456"
    source_reference="source-document001.txt"
    source_offset="789"
    source_length="456" />
</document>
```

字段含义：

| 字段 | 意思 |
|---|---|
| `this_offset` | 可疑文档里的起始字符位置 |
| `this_length` | 可疑文档里检测片段长度 |
| `source_reference` | 来源文档文件名 |
| `source_offset` | 来源文档里的起始字符位置 |
| `source_length` | 来源文档里对应片段长度 |

注意：offset 和 length 是“字符位置”，不是词数、句子数或行号。

## 2. 本项目 baseline 的思路

本项目实现的是一个完全离线、容易理解的 baseline：

1. 读取 `pairs` 文件。
2. 读取每一对 suspicious/source 文本。
3. 把文本切成句子或滑动窗口片段。
4. 用 `sentence-transformers` 的 `all-MiniLM-L6-v2` 把每个片段变成向量。
5. 计算 suspicious 片段和 source 片段之间的 cosine similarity。
6. 如果相似度超过阈值，比如 `0.80`，就认为这两个片段可能是剽窃对齐。
7. 把相邻命中片段合并。
8. 输出 PAN 风格 XML。

这个 baseline 不使用 PyTerrier、不需要 Java、不调用外部 API。Docker 镜像会在 build 阶段下载模型，运行测评时可以离线。

## 3. 项目文件说明

核心文件：

```text
main.py              # 真正运行测评的入口
requirements.txt     # Python 依赖
Dockerfile           # 提交/运行用 Docker 环境
README.md            # 项目快速说明
docs/BEGINNER_PROJECT_GUIDE.md  # 这份小白说明书
```

旧的 `retrieve.py`、`train.py`、`predict.py` 是之前实验遗留，不是这次 PAN text-alignment baseline 的主入口。你现在只需要关注：

```bash
python main.py /input /output
```

## 4. 第一次本地运行

### 4.1 安装环境

建议使用 Python 3.10。

```bash
cd /Users/wy/Generative-Plagiarism-Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果在 macOS 上装 PyTorch 很慢，优先使用 Docker 跑，因为 Docker 已经固定了 CPU 版依赖。

### 4.2 准备数据

把官网/Zenodo 下载的数据解压成这样的结构：

```text
dataset/
├── pairs
├── susp/
└── src/
```

如果还有 `_truths` 或 `truth` 目录，那是标准答案，只用于本地验证集测分，提交测试集时通常没有标准答案。

### 4.3 运行 baseline

```bash
python main.py dataset output
```

运行结束后看输出目录：

```bash
ls output
```

你应该能看到很多 `.xml` 文件。打开一个检查：

```bash
sed -n '1,80p' output/*.xml
```

### 4.4 调参数运行

最常用参数：

```bash
python main.py dataset output \
  --threshold 0.80 \
  --window-chars 550 \
  --step-sentences 2
```

参数怎么理解：

| 参数 | 影响 |
|---|---|
| `--threshold` | 相似度阈值。越低召回越高，但误报越多；越高误报少，但容易漏。 |
| `--window-chars` | 每个比较片段的大致字符长度。越大越适合改写，offset 越粗。 |
| `--step-sentences` | 滑动窗口步长。越小越细，但更慢。 |
| `--merge-gap` | 多近的命中会被合并成一个片段。 |
| `--format jsonl` | 输出 JSONL，方便自己调试；正式建议 XML。 |

推荐先试这三组：

```bash
python main.py dataset output-080 --threshold 0.80 --window-chars 550
python main.py dataset output-075 --threshold 0.75 --window-chars 700
python main.py dataset output-082 --threshold 0.82 --step-sentences 1
```

## 5. Docker 怎么运行？

### 5.1 构建镜像

```bash
docker build -t pan-plagiarism-baseline .
```

这个步骤需要联网，因为要下载 Python 依赖和 `all-MiniLM-L6-v2` 模型。构建成功后，模型已经在镜像里了。

### 5.2 用 Docker 跑本地数据

```bash
docker run --rm \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline
```

等价于在容器里执行：

```bash
python /app/main.py /input /output
```

### 5.3 Docker 里调参数

方式一：直接在命令后加参数：

```bash
docker run --rm \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline \
  /input /output --threshold 0.75 --window-chars 700
```

方式二：用环境变量：

```bash
docker run --rm \
  -e PAN_THRESHOLD=0.75 \
  -e PAN_WINDOW_CHARS=700 \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline
```

## 6. 如何测分数？

### 6.1 分数指标是什么？

PAN plagiarism detection/text-alignment 常见指标包括：

| 指标 | 意思 |
|---|---|
| `precision` | 你报出来的片段里，有多少是真的 |
| `recall` | 标准答案里的片段，你找回了多少 |
| `granularity` | 一个真实剽窃片段被你拆成了几个检测片段；越接近 1 越好 |
| `plagdet` | 综合分数，通常是主指标，综合考虑 F1 和 granularity |

直觉：

- 只要报很多片段，recall 可能高，但 precision 会低，granularity 也会变差。
- 只报最有把握的片段，precision 可能高，但 recall 会低。
- 好系统要“找得准、找得全、不要把一个片段切太碎”。

### 6.2 本地验证集测分

如果数据里有标准答案目录，比如：

```text
dataset_truth/
└── suspicious-document001-source-document001.xml
```

那么流程是：

1. 用 `main.py` 对验证集跑输出。
2. 用官方 evaluator 或 PAN 2015 text-alignment evaluator 对 `output/` 和 truth 目录打分。

伪命令如下，具体 evaluator 文件名以官网提供的评测代码为准：

```bash
python main.py validation-dataset validation-output
python evaluator.py --truth validation-truth --detections validation-output
```

如果你找不到 evaluator，先不要慌：TIRA 网站会在你提交软件后自动跑隐藏测试和评分。本地测分只是为了调参。

### 6.3 没有 truth 时怎么判断结果？

测试集通常没有标准答案。你只能做格式检查和人工抽查：

```bash
ls output | head
sed -n '1,80p' output/某个结果.xml
```

检查重点：

- 每个 pair 是否都有 XML。
- XML 的根节点是不是 `<document reference="...">`。
- feature 的 `name` 是否是 `detected-plagiarism`。
- offset 和 length 是否是非负整数。
- `source_reference` 是否对应 pair 里的 source 文件名。

## 7. 如何“训练”或改进模型？

这个 baseline 默认**不需要训练**。它使用已经训练好的 `all-MiniLM-L6-v2`，你要做的是“调参”和“误差分析”。

零基础推荐顺序：

### 第一步：只调阈值

```bash
--threshold 0.70
--threshold 0.75
--threshold 0.80
--threshold 0.85
```

经验：

- recall 太低：降低阈值。
- precision 太低：提高阈值。

### 第二步：调窗口大小

```bash
--window-chars 350
--window-chars 550
--window-chars 700
--window-chars 1000
```

经验：

- 改写很厉害：窗口大一点可能更稳。
- 需要精确 offset：窗口小一点。

### 第三步：调步长

```bash
--step-sentences 1
--step-sentences 2
--step-sentences 3
```

经验：

- `1` 更细、更慢、可能分数更好。
- `2` 是默认折中。
- `3` 更快，但可能漏。

### 第四步：换 embedding 模型

默认模型：

```bash
--model all-MiniLM-L6-v2
```

可以尝试：

```bash
--model sentence-transformers/all-mpnet-base-v2
--model multi-qa-mpnet-base-dot-v1
```

注意：如果换模型并用 Docker 提交，最好把模型也在 Docker build 阶段预下载进去，保证 TIRA 运行时不需要联网。

### 第五步：真正训练/微调（进阶）

如果你想做监督训练，需要用 truth XML 构造训练样本：

- 正样本：truth 标注的 suspicious/source 对齐片段。
- 负样本：同一 pair 里随机不相关的片段。
- 训练目标：让正样本向量更相似、负样本向量更不相似。

但这属于进阶内容。对第一次参赛来说，先把 baseline 跑通、提交成功、能测分，收益最大。

## 8. 如何提交到网站/TIRA？

PAN 通常使用 TIRA 提交软件，而不是只上传结果文件。也就是说，你要提交的是“能运行的 Docker/代码系统”，网站会把隐藏测试集挂载到 `/input`，让你的程序把结果写到 `/output`。

### 8.1 注册与进入任务

1. 注册 TIRA 账号。
2. 进入 PAN 对应年份的 Generated Plagiarism Detection 任务页面。
3. 申请/加入任务。
4. 查看任务页面给出的数据、baseline、submission 命令。

### 8.2 提交前必须满足

你的程序必须能这样运行：

```bash
python main.py /input /output
```

Docker 里也必须能这样运行：

```bash
docker run --rm \
  -v /some/input:/input:ro \
  -v /some/output:/output \
  pan-plagiarism-baseline
```

输出目录 `/output` 必须包含 XML 结果文件。

### 8.3 TIRA dry-run

具体命令以任务页面为准，常见形式类似：

```bash
tira-cli code-submission \
  --path . \
  --task generated-plagiarism-detection \
  --dataset validation \
  --command 'python /app/main.py $inputDataset $outputDir' \
  --dry-run
```

如果任务环境要求固定 `/input` 和 `/output`，命令也可能写成：

```bash
tira-cli code-submission \
  --path . \
  --task generated-plagiarism-detection \
  --dataset validation \
  --command 'python /app/main.py /input /output' \
  --dry-run
```

`--dry-run` 的意思是先测试，不正式提交。dry-run 通过后，再去掉 `--dry-run` 正式提交。

### 8.4 正式提交

```bash
tira-cli code-submission \
  --path . \
  --task generated-plagiarism-detection \
  --dataset validation \
  --command 'python /app/main.py $inputDataset $outputDir'
```

提交后，TIRA 会：

1. 构建或运行你的软件。
2. 把测试集放到输入目录。
3. 执行你的命令。
4. 收集 `/output`。
5. 用官方 evaluator 打分。

如果失败，优先看日志里是否有：

- 找不到 `pairs`。
- 找不到 `susp/` 或 `src/`。
- 模型下载失败。
- 输出目录没有 XML。
- XML 格式不对。
- 程序超时或内存爆了。

## 9. 一天的实际工作流

建议每天按这个顺序做：

1. 拉最新代码/确认当前版本。
2. 在小数据上跑一遍：

```bash
python main.py small-dataset output-test
```

3. 确认 XML 有输出。
4. 在 validation 上跑 2 到 4 组参数。
5. 如果有 evaluator，就记录分数。
6. 人工抽查几个 XML。
7. 选最好参数写入 Docker 环境变量或提交命令。
8. Docker 本地跑一遍。
9. TIRA dry-run。
10. dry-run 通过后正式提交。

## 10. 常见问题

### Q1：为什么我运行很慢？

因为每个 pair 都要算很多片段向量相似度。可以：

- 增大 `--step-sentences`。
- 增大 `--min-chars`。
- 减小 `--window-chars`。
- 减小 `--top-k`。

### Q2：为什么一个 XML 没有 feature？

说明这一对文档没有任何片段超过阈值。可以降低：

```bash
--threshold 0.75
```

### Q3：为什么检测片段太长？

减小：

```bash
--window-chars 350
--merge-gap 50
```

### Q4：为什么检测片段太碎？

增大：

```bash
--merge-gap 200
--window-chars 700
```

### Q5：TIRA 上不能联网怎么办？

Dockerfile 已经在 build 阶段下载并保存模型：

```dockerfile
SentenceTransformer("all-MiniLM-L6-v2").save("/models/all-MiniLM-L6-v2")
```

运行时默认使用：

```text
PAN_MODEL=/models/all-MiniLM-L6-v2
```

所以运行阶段不需要联网。

## 11. 最小提交检查表

提交前逐项确认：

- [ ] `python main.py /input /output` 能运行。
- [ ] Docker 能 build 成功。
- [ ] Docker 能在 toy 数据上输出 XML。
- [ ] XML 文件名是 `suspicious-stem-source-stem.xml`。
- [ ] XML 根节点是 `<document reference="suspicious-file.txt">`。
- [ ] `feature name="detected-plagiarism"`。
- [ ] offset/length 都是整数。
- [ ] 没有依赖 Java、PyTerrier、外部 API。
- [ ] 模型已经 baked into Docker image。
- [ ] TIRA dry-run 通过后再正式提交。

## 12. 参考链接

- PAN 2025 Generated Plagiarism Detection task page: https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection
- PAN 2025 dataset record: https://zenodo.org/records/14969012
- PAN code repository: https://github.com/pan-webis-de/pan-code/tree/master/clef25/generated-plagiarism-detection
- TIRA: https://www.tira.io/
