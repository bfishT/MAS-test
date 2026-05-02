# SWE-bench + LangGraph Agent Pipeline 工作总结

## 项目目标

实现一个自动化流水线：读取 SWE-bench 问题 → 自动构建 Docker 容器 → LLM 生成 DAG 计划 → 自动转换为 LangGraph → Agent 在代码库中执行修复 → 输出 patch → SWE-bench 评测。

## 已完成的工作

### 1. 环境搭建

- **LangGraph 安装**：在 `dataset/` 子项目的 uv 虚拟环境中安装了 `langgraph>=1.1.9`、`docker>=7.1.0`、`swebench`
- **依赖配置**：更新了 `dataset/pyproject.toml`，添加了 langgraph、docker、swebench 依赖

### 2. 核心脚本 `dataset/run_with_langgraph.py`

当前版本是 **Docker + 自动容器管理** 模式：

**流程：**
1. 从 `swebench_dataset.json` 加载 SWE-bench 实例
2. 自动构建 Docker 容器（base 镜像 → env 镜像 → instance 镜像 → 启动容器）
3. 调用 LLM 生成 DAG 格式的修复计划
4. 解析 DAG 文本为节点和边（容错解析，失败时 fallback 为单节点）
5. 构建 LangGraph StateGraph：
   - 每个 DAG 节点 = 一个带工具的 Agent
   - 并行节点由 LangGraph 自动并行执行
   - Agent prompt 中注入 `FAIL_TO_PASS` 测试信息，鼓励验证
6. Agent 通过 `container.exec_run` 在 Docker 容器内操作
7. 执行完毕后 `git diff` 提取 patch
8. 自动清理容器
9. 输出 `predictions.jsonl`（SWE-bench 标准格式）

**Agent 工具集（7个）：**
- `search_code`：grep 搜索代码
- `read_file`：读取文件内容
- `edit_file`：部分替换文件内容（新增，避免全量覆写）
- `write_file`：全量写入文件（保留，用于创建新文件）
- `list_directory`：列出目录
- `run_test`：运行指定测试用例（新增，用于验证修复）
- `run_command`：执行任意命令

**命令行参数：**
- `--dataset`：SWE-bench 数据集路径（默认 swebench_dataset.json）
- `--model`：模型名（默认 glm-5.1）
- `--api-key` / `--base-url`：LLM API 配置
- `--run-id`：容器命名后缀（默认 langgraph_agent）
- `--container`：指定已有容器名/ID（可选，跳过自动构建）
- `--keep-container`：Agent 跑完是否保留容器（默认自动清理）
- `--start-index` / `--end-index`：跑哪些实例

### 3. Docker 构建修复

**修改了 `swebench/harness/dockerfiles/python.py`**：
- 在 base 镜像 Dockerfile 中加入阿里云 apt 镜像源替换
- 解决了国内环境 `apt update` 失败的问题

### 4. 遇到的问题及解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| LangGraph 并行节点报 `InvalidUpdateError` | 并行节点同时写 state | 用 `Annotated[dict, merge_dicts]` 和 `Annotated[str, keep_last]` |
| 本地 workspace 空，Agent 找不到代码 | 用 questions 文件没有 repo 信息 | 改用 `--dataset swebench_dataset.json` |
| Agent 反复 pip install 浪费轮次 | 本地没有项目构建环境 | 改用 Docker 容器方案 |
| Docker apt update 返回 100 | 国内网络问题 | Dockerfile 换阿里云源 |
| `make_test_spec` 断言失败 | `base_image_tag=None` | 传 `instance_image_tag="latest", env_image_tag="latest"` |
| SWE-bench logger 没有 `log_file` 属性 | 用了标准 `logging.Logger` | 改用 swebench 的 `setup_logger` |
| DockerHub 远程镜像不可用 | 需要认证或不存在 | 改为本地构建 + 换源 |
| DAG 解析失败导致流水线中断 | LLM 输出格式不稳定 | 容错解析 + fallback 为单节点 DAG |
| Agent 用 write_file 全量覆写破坏代码 | 没有部分编辑工具 | 新增 `edit_file` 工具 |
| Agent 改完代码不验证 | 缺少测试工具 | 新增 `run_test` 工具 + FAIL_TO_PASS 注入 prompt |
| 需手动构建容器 | 旧版需 --container 参数 | 改为自动构建/清理容器 |
| 并行节点修改同一文件冲突 | merge_dicts 覆盖 | planner prompt 中限制"不同节点不修改同一文件" + merge 冲突日志 |

## 当前状态

脚本已就绪，**全自动模式**：自动构建容器 → Agent 执行 → 自动清理 → 输出 patch。

## 使用方法

### 跑 Agent（全自动）

```bash
cd ~/workspace/SWE-bench/SWE-bench/dataset

uv run python run_with_langgraph.py \
  --api-key YOUR_KEY \
  --base-url https://opencode.ai/zen/go/v1 \
  --model glm-5.1 \
  --start-index 0 --end-index 1
```

脚本会自动：构建镜像 → 启动容器 → 生成 DAG → Agent 执行 → 提取 patch → 清理容器

### 连接已有容器（可选）

```bash
uv run python run_with_langgraph.py \
  --api-key YOUR_KEY \
  --base-url https://... \
  --container <已有容器名> \
  --keep-container \
  --start-index 0 --end-index 1
```

### 评测

```bash
cd ~/workspace/SWE-bench/SWE-bench

# 用本地数据集路径（推荐）
python -m swebench.harness.run_evaluation \
  --dataset_name dataset/swebench_dataset.json \
  --predictions_path dataset/predictions.jsonl \
  --max_workers 4 \
  --run_id glm5_dag_test

# 或用 HuggingFace 数据集名（需确保 instance_id 在该数据集中）
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path dataset/predictions.jsonl \
  --max_workers 4 \
  --run_id glm5_dag_test
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `dataset/run_with_langgraph.py` | 主脚本：DAG 生成 + LangGraph Agent 执行 |
| `dataset/generate_plans.py` | 原始 DAG 生成脚本（仅生成，不执行） |
| `dataset/plans/visualize_dag.py` | DAG 可视化脚本 |
| `dataset/pyproject.toml` | 子项目依赖配置 |
| `dataset/swebench_dataset.json` | SWE-bench 数据集（2294 条） |
| `dataset/predictions.jsonl` | Agent 输出的 patch 文件 |
| `swebench/harness/dockerfiles/python.py` | 修改了 apt 源的 Dockerfile 模板 |

## 可优化方向

1. **DAG planner 可换模型**：当前用同一个模型做规划和执行，可分开
2. **Agent 工具增强**：加 `find_file`、`git_log` 等
3. **节点间信息传递优化**：当前传文字摘要，可传关键文件路径/行号
4. **重试机制**：验证失败时自动回退 git 修改重试
5. **并行度控制**：可限制同时运行的 Agent 数量避免 API 限流
6. **批量运行**：先批量构建所有镜像，再批量跑 Agent，减少等待时间
