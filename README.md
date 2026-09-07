# 上下文场景感知增强

## v3.5.0

新增“先发图、后提问”的上下文看图能力：明确场景自动带入近期单图，复杂指代通过工具按需读取。图片缓存有界，自动图片临时注入，工具图片在下一轮退出视觉历史。详见下方“上下文按需看图”。

### 保留的图片兼容能力（v3.4.x）

- 历史图片预处理后保持为自包含 `data URI`，不会把短期缓存路径写回持久化会话。
- 自动移除会话中已经失效的本地图片引用，单张过期图片不再拖垮整轮 LLM 请求。
- 图片组件使用正确的 `file URI`/本地 `path` 语义，并复用规范化路径避免重复压缩。
- 持久化会话历史中的图片也会进入 LLM 请求图片预处理，不再只处理当前消息和引用消息。
- GIF 会在单次 LLM 请求中提取首帧为 PNG 临时副本，兼容不支持 `image/gif` 的模型。
- 引用消息中的图片文件会按真实内容归一化为 `Image`, 避免 Core 再次回查 OneBot。
- 自动支持 Pillow 可识别的 PNG、JPEG/JPG、WebP、GIF、BMP、TIFF、ICO 等图片格式。
- 伪图片后缀和损坏文件保持原 `File`, 不会阻断正常消息处理。
- 新增可选的 LLM 请求图片压缩, 仅生成临时副本, 不修改原图。
- 支持按文件大小或最长边触发, 并可调 JPEG 质量、最低质量、目标输出大小和输入上限。
- QQ CDN 图片下载增加完整性校验、指数退避重试和短时失败缓存, 减少引用大图时的下载中断。
- 修复 `/new` 和 `/reset` 后插件上下文未清理的问题, 覆盖第三方 Agent runner。
- 兼容 AstrBot 当前的 `_clean_group_context_session` 信号, 并保留旧版 `_clean_ltm_session` 兼容。
- 兼容 `astrbot_plugin_cmdmask` 的伪装指令, 根据插件提供的真实 target 清理上下文。

**让你的 Bot 在群聊中不再"抢答"别人的问题**

---

## 上下文按需看图

群友先发图片，下一条消息再问 Bot“图里是什么”，不必把图片重新发一遍。插件平时只提供图片的发送者、时间、消息 ID 和位置；真正需要时才把选中的图片交给视觉模型。

- **直接接话**：同一发送者在 90 秒内发过唯一单图，随后明确问图时，自动临时带入该图。当前消息已有图片、引用、多人图片指代不明、请求比较多图或主动搭话时不自动选图。
- **按需回看**：模型通过 `context_aware_view_images` 选择当前会话索引中的 ID；支持一次查看多张、`auto` 概览和 `high` 查看细节。像“这啥”“小王刚才那张截图怎么了”等表达由模型结合索引决定是否调用工具，不保证每种表达都触发自动带图。
- **短期缓存**：默认每会话 20 张、30 分钟，所有会话共用 64 MB 图片与来源预算。图片索引独立于文字消息窗口；后台预取最多 2 个并发、16 个待处理任务，队列繁忙时保留来源供按需加载。下载和解码工作内存不包括在 64 MB 中。
- **避免反复带图**：自动图片使用临时内容标记。工具图片在当前 Core 中可能保存至本轮历史；清理依赖内置 Agent runner 的工具图片标记与相邻图像结构，第三方 runner 需单独验证；插件会在下次请求前精确移除本工具追加的历史图像，只保留文字与工具调用关系，不清理用户原生图或其他工具图。
- **隔离与清空**：图片 ID 仅能在当前请求的会话快照使用；其他群、新到消息不会串入。`reset`、`new`、会话切换清理信号、插件重载或 Bot 重启会使旧索引失效。

使用此功能需要当前聊天模型支持 **image**；按需工具还需要 **tool_use**，且 AstrBot 工具列表中的 `context_aware_view_images` 已启用。无需开启 `image_caption`，普通收图不会额外调用模型。模型没有视觉能力时，本功能不会生成虚构的图片描述。

`auto` 的最长边为 1600，`high` 最长边为 4096，均为有大小限制的 JPEG 视觉副本；原消息和原图不修改，透明图片以白色背景合成。需要原始像素或精确小字时，请直接发送/引用原图。沿用 `show_recent_images_allow_gif` 设置；混有被过滤 GIF 的整条消息不加入回看索引，允许时仅查看首帧，不支持动态理解。图片输入上限沿用 `image_download_max_bytes` 并受内存预算约束，像素上限为 2500 万。

图片只保留在短期缓存，过期、清空、重启或缓存压力淘汰后可能需要重发。预取不保证外部链接永久可用；只允许公网 HTTP(S) 图片下载，私网图片服务不会被回看工具访问。自动带图最多等待 2 秒，未准备好时交给按需工具；工具下载有超时与失败退避。

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `image_recall` | `true` | 开启图片索引和按需工具 |
| `image_recall_auto` | `true` | 高置信提问自动临时带图 |
| `image_recall_ttl` | `1800` | 保留秒数 |
| `image_recall_per_session` | `20` | 每会话图片上限 |
| `image_recall_memory_mb` | `64` | 图片数据与来源缓存预算 |
| `image_recall_max_per_request` | `2` | 每轮不同图片上限，包含自动图片 |

### 开发验证

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
ruff check .
```

在已安装 AstrBot 的 Python 环境中，可另行执行 `python tests/verify_astrbot_integration.py`（AstrBot 源码需在 `PYTHONPATH`）。该验证使用真实消息类型和 Agent tool loop，以及替代 Provider，不产生真实模型费用或群消息；真实模型与平台收发应另行验证。

## 版本要求

需要 AstrBot `>=4.26.8`（本次集成验证版本）。本插件会把场景提示标记为临时内容，避免动态上下文被写入会话历史。

---

# ⚠️ 安装后必须配置

## 🚨 必须关闭框架内置的「群聊上下文感知」

**安装本插件后，如果不关闭框架内置功能，会导致：**
- LLM 收到两份群聊记录
- 上下文快速膨胀
- Token 消耗翻倍
- 可能触发上下文溢出

### 关闭方法

**方法一：管理面板**
```
提供商设置 → 群聊上下文感知(原聊天记忆增强) → 关闭「群聊上下文感知增强」开关
```

**方法二：配置文件**
```yaml
provider_ltm_settings:
  group_icl_enable: false  # 必须设为 false！
```

---

## 推荐完整配置

```yaml
# AstrBot 框架配置
provider_ltm_settings:
  group_icl_enable: false  # 关闭框架内置群聊记录
  active_reply:
    enable: true           # 主动回复功能可以保留

provider_settings:
  max_context_length: 20   # 限制对话轮次，防止无限增长
```

---

## 推荐联动插件

以下插件都不是本插件的硬依赖，可以单独安装和使用。组合后可以补足消息真实性判断和自主沉默能力，让群聊上下文不仅完整，而且更可信、更克制。

### [消息真实性校验](https://github.com/muyouzhi6/astrbot_plugin_message_authenticity)

- 识别普通文本伪造的 `[图片]`、`[红包]`、`[转账]` 等标签，并校验 OneBot/NapCat 上报的真实结构化消息。
- 检测到真实红包或转账事件时，会将可读描述写入 Context Aware 会话，让 Bot 在后续对话中知道“谁、何时、发了什么”。
- 该联动是软依赖；未安装 Context Aware 时，消息真实性校验仍可独立工作。
- 真实红包事件只代表平台上报了可信的钱包结构，不代表 Bot 已领取红包或资金已经到账。

### [算了不说了](https://github.com/muyouzhi6/astrbot_plugin_suanle_bushuo)

- Context Aware 提供完整的群聊场景和对话对象判断，`算了不说了` 提供 `keep_silent` 工具，让 LLM 在不该插话或无需回复时主动保持沉默。
- 支持黑名单强阻断，同时仍将被阻断用户的消息作为 `<blocked_messages>` 临时上下文提供给 LLM，避免群聊信息断裂。
- 黑名单上下文由 `算了不说了` 独立维护，不调用 Context Aware 的清理接口，两个插件各自负责自己的数据边界。
- `keep_silent` 依赖模型的 function-calling/tools-use 能力；不支持工具调用的模型无法保证自主沉默。

---

## 功能对比

| 功能 | 本插件 | 框架内置 LTM |
|------|--------|-------------|
| 记录群聊消息 | ✅ 50条（可配置） | ✅ 300条 |
| **分析对话对象** | ✅ 核心功能 | ❌ |
| **触发类型识别** | ✅ | ❌ |
| **行为指导** | ✅ | ❌ |
| 图像转述 | ✅ 可选 | ✅ |
| LLM 请求图片压缩 | ✅ 可选、可调 | ✅ 全局配置 |
| 注入位置 | 用户消息 | 系统提示词 |

**本插件可完全替代框架内置 LTM 的群聊记录功能，且功能更强大。**

---

## 解决什么问题？

你的 Bot 是不是经常这样：

- 小明问小红："你吃饭了吗？" → Bot 抢答："我还没吃呢！"
- 群友们在聊天 → Bot 主动插话，还以为别人在问它
- 被主动回复触发后，把所有问题都当成问自己的

**这个插件就是来解决这个问题的。**

---

## 效果

安装后，Bot 会收到这样的场景提示：

```xml
<conversation_scene>
  <trigger type="active">你是主动加入这个对话的，没有人在叫你</trigger>
  <current_message>
    <sender>小明</sender>
    <talking_to>小红</talking_to>
    <content>你吃饭了吗？</content>
  </current_message>
  <instruction>【重要】你是主动加入对话的！小明 正在和 小红 对话，不是在问你。</instruction>
</conversation_scene>
```

| 场景 | 以前 | 现在 |
|------|------|------|
| A 问 B 问题 | Bot 抢答 | Bot 知道不是问自己 |
| @Bot | 正常回复 | 正常回复 |
| 回复 Bot 消息 | 正常回复 | 正常回复 |
| 主动触发 | 以为都在问自己 | 清楚知道谁在和谁说话 |

---

## 插件配置项

### 基础配置

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `enable` | 启用插件 | `true` |
| `bot_names` | Bot 的昵称列表（用于检测被提及） | `[]` |
| `max_history` | 每个群保留的消息数 | `50` |
| `max_groups` | 最大缓存群数（LRU 淘汰） | `100` |
| `dialogue_window` | 注入 LLM 的对话流条数 | `8` |
| `enable_dialogue_flow` | 显示对话流（谁→谁） | `true` |
| `only_group_chat` | 仅群聊生效 | `true` |
| `warn_builtin_ltm` | 检测到内置群聊上下文感知时输出警告 | `true` |
| `show_recent_images` | 单独列出最近图片消息，避免图片上下文淹没在普通对话流里 | `true` |
| `image_context_window` | 从最近 N 条消息里提取图片并单独注入 | `20` |

### 图像转述配置

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `image_caption` | 启用图像转述 | `false` |
| `image_caption_lazy` | 延迟图像转述，仅在生成 LLM 回复时转述上下文窗口内的图片 | `false` |
| `image_cache_dir` | 图片本地缓存目录；留空使用 `plugin_data/astrbot_plugin_context_aware/cached_images` | `""` |
| `image_cache_ttl` | 图片缓存文件保留时间；启动时、后台任务及运行中下载图片前清理过期文件 | `3600` |
| `image_caption_provider_id` | 图像转述提供商（下拉选择） | 默认提供商 |
| `image_caption_prompt` | 图像转述提示词 | `请用中文简洁描述...` |

**注意**：启用图像转述后，每张图片会调用一次 LLM，会产生额外费用和延迟。

`show_recent_images_allow_gif` 控制 `<recent_images>` 场景区块、图像转述及按需看图索引中的 GIF，不会删除 AstrBot Core 已持久化的会话图片。启用 `llm_image_compress` 后，历史 GIF 会在发送给模型前转换为首帧 PNG，并以自包含 `data URI` 进入本次请求；已经失效的历史本地图片引用会被安全移除，不影响同轮文本和其他图片。

### LLM 请求图片压缩

`llm_image_compress` 默认关闭。开启后, 插件只在图片即将进入 LLM 请求时生成临时压缩副本, 原始图片和 Bot 生成的 4K 文件不会被修改。该功能独立于场景上下文开关, 私聊或关闭 `enable` 时仍可单独使用。

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `llm_image_compress.enable` | 启用 LLM 请求图片压缩 | `false` |
| `llm_image_compress.min_size_mb` | 文件达到该大小时触发压缩 | `4.0` |
| `llm_image_compress.max_edge` | 压缩后最长边 | `2048` |
| `llm_image_compress.quality` | JPEG 首选质量 | `90` |
| `llm_image_compress.min_quality` | JPEG 自适应最低质量 | `75` |
| `llm_image_compress.max_output_size_mb` | 单图目标输出大小 | `2.0` |
| `llm_image_compress.max_input_size_mb` | 允许处理的单图大小上限 | `50.0` |
| `llm_image_compress.download_retries` | 远程图片下载尝试次数 | `3` |
| `llm_image_compress.download_timeout` | 单次远程下载超时秒数 | `15` |

处理规则:

- 引用链中的图片 `File` 会在 Core 构建请求前按真实内容识别并转换, 不依赖扩展名。
- 文件达到体积阈值, 或最长边超过 `max_edge` 时才压缩。
- 普通静态图输出 JPEG; 带透明通道的图片保持 PNG; GIF 提取首帧并输出 PNG, 其他动图保持原样。
- 输出超过目标大小时, 先逐步降低 JPEG 质量, 再逐步缩小分辨率。
- 下载或压缩失败时保留原引用, 不会阻断正常对话。

### 图片上下文增强

即使群友发送图片时没有触发 Bot，本插件也会记录图片消息。后续触发 LLM 时，会从最近 `image_context_window` 条消息里提取图片，单独进入 `<recent_images>` 区块：

```xml
<recent_images>
  <image sender="小明" talking_to="群">[图片: 一张会议白板照片，写着项目排期]</image>
</recent_images>
```

如果未启用图像转述，图片会使用 AstrBot 的消息概要或 `[图片]` 占位进入上下文；如果启用图像转述，则会附带视觉模型生成的图片描述。

---

## 技术特点

- **内存安全**：LRU 淘汰机制，严格限制消息和会话数量上限
- **高效稳定**：纯规则分析，图像转述为可选功能
- **无侵入性**：场景注入不覆盖框架提示; 图片压缩只替换本次 LLM 请求使用的临时副本
- **可持久运行**：会话、图片描述和下载缓存均有容量或 TTL 清理机制

---

## 原理

1. 监听所有群消息，分析 @、回复、上下文推断出"谁在和谁说话"
2. 在 LLM 请求前注入结构化场景描述
3. 告诉 LLM：这条消息是对谁说的、你是被叫的还是主动插话的
4. (可选) 将群友发送的图片转为文字描述
5. (可选) 在图片进入 LLM 请求前生成受控大小的临时压缩副本

---

**让 Bot 学会"察言观色"，不再尬聊。**

---

本插件开发QQ群：215532038

<img width="400" alt="QQ群二维码" src="https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee" />
