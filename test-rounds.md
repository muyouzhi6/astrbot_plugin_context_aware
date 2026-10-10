# 测试记录

## Fix round 3：Windows 适配器文件路径（2026-10-11）

版本保持 3.10.0。本轮针对独立审查指出的 `file://C:\图片#1.jpg` 截断问题，先增加回归用例，再修改 `_local_image_path`：盘符加反斜杠的适配器路径保留未编码的 `#` 和 `?`；标准 `file:` URI 继续解析查询与片段，`%23` 仍解码一次。

| 阶段 | 日志 | 结果 |
| --- | --- | --- |
| 修复前复现 | [回归失败](logs/round3-red-regressions.log) | 5 项测试运行，4 个失败、2 个错误，均指向未编码分隔符导致的路径截断或图片读取失败。 |
| 定向复测 | [Python 3.13](logs/round3-focused-python313.log)、[Python 3.10](logs/round3-focused-python310.log)、[图片链路](logs/round3-pipeline-python313.log) | 两种解释器的 Windows 回看用例各 5 项通过；现有图片压缩、缓存和历史链路 11 项通过。 |
| 全量测试 | [3.13 unittest](logs/round3-python313-unittest.log)、[3.13 pytest](logs/round3-python313-pytest.log)、[3.10 unittest](logs/round3-python310-unittest.log)、[3.10 pytest](logs/round3-python310-pytest.log) | 两种解释器各 165 项 unittest、320 项 pytest 与 72 个子场景通过，无失败或跳过。pytest 使用既有 `ASTRBOT_TEST_PYTHON` 环境。 |
| 静态检查 | [Ruff](logs/round3-ruff.log)、[3.13 compileall](logs/round3-compile-python313.log)、[3.10 compileall](logs/round3-compile-python310.log)、[diff check](logs/round3-diff-check.log) | 全部退出码为 0；无编译错误或空白字符错误。 |

评分：**9.4/10**。目标输入先复现再通过，编码字符与标准 URI 语义均有断言，两个 Python 版本的全量检查通过。Windows 文件打开在 Linux 上模拟；AstrBot 4.26.8 实机验证由协调方执行，本轮记录不包含该结果。

本轮未连接 SSH、未推送、未发表评论。改动均位于仓库目录。
