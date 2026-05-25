目前 Agent 类的工具层出不穷，从 Claude Code、Codex 到 Cursor、Manus。这些 Agent 工具都是将 LLM 作为调度的核心，用来接受自然语言指令，完成对应的任务。LLM 的主要作用就是理解语义，并且把语义拆分为对应可执行的小任务。

在 Agent 的核心流程中，需要多轮调用 LLM 的 API，直到任务完成。如果任务足够简单，与 LLM 的交互可以在一轮中完成。

但当前 Agent 承担的任务越来越复杂，无法在一轮 API 调用中完成。在理解任务的过程中，需要额外的信息输入，LLM 本身没有获取外部信息的能力，只能通过多轮对话从外部收集。

Agent 在执行任务的过程，是 LLM 不断从外部收集信息，做决策的过程。这篇文章就来详细聊一聊 Agent 实现中，怎么与 LLM 做交互。

## 1. Agent 是什么

按照 Anthropic 的定义，Agent 就是 LLM 在循环中动态决定使用工具与流程的系统。这个定义中有两个关键，一个是循环，一个是动态决定使用工具和流程，这也是 Agent 与 Workflow 最大的区别。

那么按照定义，一个 LLM Agent 在实现上可以简化为下面的这段代码：
```python
while True:
    response = llm.chat(messages, tools)         # ① 问模型
    if response.tool_calls:                       # ② 模型要调工具
        for tc in response.tool_calls:
            result = run_tool(tc.name, tc.args)   # ③ 你的代码执行
            messages.append(tool_result(result))  # ④ 把结果塞回上下文
        continue
    if response.content:                          # ⑤ 模型给了最终答复
        print(response.content)
        break
```


我们可以发现，LLM 本身不执行任何代码和逻辑，它只输出一段 JSON，里面写明工具名和参数。然后让 Agent 的执行环境去调用这些工具拿到结果，将响应附加到调用 LLM API 的参数中，等待 LLM API 的下一轮响应。

这就是 Agent 的核心流程，其他更复杂的逻辑都是围绕这个核心流程建立的。

### 1.1 执行流程总览

把 Agent 循环简化成一张图片。①–④ 是准备阶段（初始化、注册工具、组装上下文）。⑤ 是每一轮和模型的对话。模型返回后落到 A/B 两个分支。A 调工具回填后继续循环，B 直接退出。

<img width="692" height="536" alt="Clipboard_Screenshot_1779720285" src="https://github.com/user-attachments/assets/3283e0b7-3460-43ef-acd9-919828ac46e9" />


### 1.2 最小可运行示例

下面是一个最简单的 Agent 示例，用于获取一个地域的天气情况，例子虽然简单，但是却包含Agent 运行最核心的要素。

在 Agent 运行过程中，需要将外部的信息按照一定的规则传递给 LLM，下面的例子中使用的是 OpenAI 的 chat.completions API，这也目前与 LLM 交互用的最多的方式。虽然有些 LLM 有自己的交互 API，但大体上逻辑是一致的。

下面的代码逻辑很简单，核心就是向 LLM 提交任务和可用的 tools，让 LLM 来决定调用哪些 tool，并且持续向 LLM 发送 tool 调用的结果，直到 LLM 产出最终的结果。

这里多说一句，由于 LLM 是通过 NTP（**Next Token Prediction**）来产生内容，当下一个词是结束符的时候，LLM 的内容产出也就结束了。

```python
from openai import OpenAI
import json

client = OpenAI(base_url="https://api.moonshot.cn/v1", api_key="sk-...")

# 1. 本地技能（真正干活的函数）
def get_weather(location):
    if "深圳" in location:
        return json.dumps({"temperature": "15C", "condition": "大风"}, ensure_ascii=False)
    return json.dumps({"temperature": "未知"}, ensure_ascii=False)

TOOL_REGISTRY = {"get_weather": get_weather}

# 2. 工具菜单（给模型看的 schema）
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

# 3. 上下文初始化
messages = [
    {"role": "system", "content": "你是助手，问天气务必先调工具。"},
    {"role": "user", "content": "深圳现在的天气怎么样？"},
]

# 4. Agent 循环
for _ in range(5):
    resp = client.chat.completions.create(
        model="kimi-k2-0905-preview", messages=messages, tools=tools, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = TOOL_REGISTRY[tc.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })
        continue

    if msg.content:
        print(msg.content)
        break
```


那么这个 Agent 在与 LLM 交互的过程中，具体的交互细节是怎么样的？

## 2. LLM API 的协议层，消息、role、tools

要完整理解 Agent 的工作，需要先理解一下 LLM 的 API，下面以 OpenAI Chat Completions 为例，来具体看看 API 的一些细节。

在与 chat.completions API 交互时，会按照下面的方式来初始化：
```python
completion = client.chat.completions.create( 
             model="kimi-k2-0905-preview", 
             messages=[ {"role": "user", "content": "你好"} ], 
             tools = tools
             )
```
### 2.1 messages 与 role 详解

`messages` 是一个数组，记录从对话开始到现在的全部消息。它是 Agent 的工作内存。LLM 是无状态的纯函数，每次调用都要把完整历史发回去。

每条 message 有一个 `role`，role 由 LLM API 提前定义好，不能自定义。多数厂商对未知 role 直接返回 400。Chat Completions 协议下骨架由 4 个 role（system、user、assistant、tool）加 1 个顶层字段（tools）组成，下面逐一展开。

**system，开发者写下的出厂设定**

出现位置在 messages 数组最前面。内容包括身份、能力边界、回答风格、禁区。模型每次回答都把它当作前提条件参考。多数厂商允许多条 system，但合并成一条更稳，避免规则相互覆盖。

```python
{"role": "system", "content": "你是天气助手，回答前必须先查工具。"}
```

**user，终端用户的输入**

多轮对话里会反复出现，每次新提问都新增一条。可以加 `name` 字段做多用户区分。`content` 支持多模态（文本、图片），具体看模型支持什么。

```python
{"role": "user", "content": "深圳天气怎么样？"}
{"role": "user", "name": "alice", "content": "..."}
```

**assistant，模型的输出**

assistant 有两种形态。

一种是纯文本回答，`content` 非空，`tool_calls` 为 null。这是模型给出最终答复的标志。

另一种是工具调用请求，`tool_calls` 非空，`content` 通常为空字符串或 null。模型决定调用一个或多个工具时使用。

偶尔会出现两者同时存在的边走边说模式。`content` 是前言（比如"我来查一下"），`tool_calls` 是后续动作。这种情况要靠 if/elif 顺序优先看 tool_calls 来正确分支。

assistant 既是 API 当轮的输出，也要塞回 messages 做历史。下一轮模型才能看到自己之前说过什么、调过什么。

```python
# 决定调工具
{"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\":\"深圳\"}"}}
]}

# 给最终回答
{"role": "assistant", "content": "深圳 15℃ 大风..."}
```

**tool，工具执行结果**

你的代码跑完工具后写回去的结果。必须带 `tool_call_id` 字段，严格对应上一条 assistant 的某个 `tool_calls[i].id`。配对错误会被服务端直接拒收，提示 `Tool call id 'xxx' not found in preceding assistant message`。

```python
{"role": "tool", "tool_call_id": "call_1",
 "name": "get_weather",
 "content": "{\"temp\": 15, \"condition\": \"大风\"}"}
```

**一次完整对话的形态**

```python
messages = [
    {"role": "system", "content": "你是天气助手，回答前必须先查工具。"},
    {"role": "user", "content": "深圳和上海哪个更暖？"},
    {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"深圳\"}"}},
        {"id": "call_2", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"上海\"}"}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":15}"},
    {"role": "tool", "tool_call_id": "call_2", "content": "{\"temp\":18}"},
    {"role": "assistant", "content": "上海更暖，18℃ 比深圳 15℃ 高。"},
]
```

五种成员分工清楚。system 定身份，user 提需求，assistant 决策与回答，tool 提供外部数据回流，tools 描述可调用的能力。每一层职责清晰，不会越权。

要做研究员、批评家这类自定义身份，在 system 里描述，或用 `name` 字段标注（`{"role": "user", "name": "alice"}`）。不要在 role 上自创字符串。

### 2.2 role 在不同厂商 LLM API 的实现差异

Chat Completions 是事实标准，但具体到每家 LLM API，role 的命名、位置、配对规则都有微妙差异。下面分四类典型实现看清楚。

**OpenAI 自家**

支持完整四角色 `system`、`user`、`assistant`、`tool`。多条 `system` 合法但建议合并。role 字符串严格枚举，未知值直接 400。工具调用走 `assistant.tool_calls[]` 加 `role="tool"` 回填。同 role 允许连续出现，虽然不建议。多模态走 `content` 数组，每个元素带 `type` 字段。

**Anthropic（Claude）**

只有两个 role，`user` 和 `assistant`。没有 `system` role，对应字段是 API 顶层的 `system` 参数。没有 `tool` role，工具调用和结果都打包进 `content` block。工具调用是 `assistant.content` 里的 `{"type": "tool_use", "id": "...", "input": {...}}`。工具结果是 `user.content` 里的 `{"type": "tool_result", "tool_use_id": "...", "content": "..."}`。messages 必须严格 user/assistant 交替，不能连续两个同 role。思维链是原生 `{"type": "thinking"}` block，不是扩展字段。

**Google Gemini**

用 `user` 和 `model` 两个 role，不叫 assistant。系统指令叫 `systemInstruction`，是顶层参数。工具调用通过 `parts[]` 数组中的 `functionCall` 字段。工具结果用 `role: "function"` 加 `functionResponse` 字段回填。`parts` 数组允许文本与函数调用混排。历史消息严格 user/model 交替。

**国产模型与本地推理引擎**

国产模型（Moonshot Kimi、DeepSeek、智谱 GLM、阿里 Qwen 等）几乎全部兼容 OpenAI Chat Completions 协议，包括四角色命名、`tool_calls` 字段、`tool_call_id` 配对规则。但有几个常见限制：
- 部分模型早期不支持多条 `system`，要在客户端合并
- 智谱 GLM 早期版本要求 `system` 必须是 `messages[0]`
- Qwen 在 OpenAI 协议之外加了 `partial` 等扩展字段，做流式生成续写

本地推理引擎（Ollama、vLLM、llama.cpp、LM Studio）通常都实现 OpenAI Chat Completions 兼容层。但底层是模型自己的 chat template 决定 role 怎么编码：
- Llama 3 chat template 把 `tool` role 编码为特定 special token
- Qwen2 chat template 用 `<|im_start|>tool` 包裹工具结果
- 自定义模型如果没有原生 tool role 训练，需要在 prompt 层面用文本约定模拟

所以同一份 messages 数组发给不同后端，落到模型实际输入的字符序列并不完全相同。这也是 Chat Completions 协议层面统一、token 层面有差异的原因。

### 2.3 tools，给模型看的工具菜单

tools 不在 messages 数组里，而是请求体顶层的独立字段。它是一个 JSON Schema 数组，每一项描述一个可调用的函数。函数名、参数结构、参数类型都写在里面。模型在生成下一条 assistant 之前会先看完 tools，再决定要不要调工具、调哪个、传什么参数。一般会把工具调用产生的结果的 role 设置为 tool。

```python
client.chat.completions.create(
    model="...",
    messages=[...],          # 对话历史
    tools=[...],             # 可用工具说明书，与 messages 同级
    tool_choice="auto",      # 调度策略
)
```

把能力描述放在 messages 之外是 Chat Completions 一个有意的设计选择。对话内容和能力清单分层，每条消息的语义都能被服务端、客户端、第三方代理（LiteLLM、OpenRouter 等）独立解析、缓存、降级。


```python
tools = [{
    "type": "function",                            # ← 预留扩展位
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {                            # ← JSON Schema
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]
```

注意一个常被误解的点。`type` 字段是协议预留位，但能用哪些 type 由后端模型决定。对于 OpenAI Chat Completions 和所有国产模型，实际能用的只有 `"function"`。`web_search`、`code_interpreter` 这些内置工具 type 仅在 OpenAI Responses API 等更新的接口里合法。

`tool_choice` 控制策略。
- `auto`（默认），模型自己决定调不调
- `none`，禁用工具
- `required`，强制必须调一个工具
- `{"type":"function", "function":{"name":"..."}}`，指定调某个

**和 Structured Output 的区别**

tools 解决的是模型决定调哪个外部能力、参数是什么。但有时候开发者只想让模型按某个 schema 直接吐一份 JSON，不需要再执行什么工具。比如把简历抽出 `{name, email, skills[]}`、给一段文本打分类标签、把用户描述转成数据库查询参数。这是 Structured Output 的场景。

直接让模型「输出 JSON」存在两个老毛病：一是格式跑偏，模型可能输出 markdown 包裹的代码块、末尾多个逗号、前后多一句客套话；二是字段漂移，让你给 `{"age": int}`，模型可能给成字符串 `"25 岁"`，或者多给一个你没要求的字段。

Structured Output 从协议层一次性解决这些问题，**保证输出可以被 `json.loads` 直接解析、字段结构与 schema 完全一致**。底层不是「让模型加倍努力按格式输出」，而是**约束解码**——每生成下一个 token 时，先把不符合 schema 的 token 概率置零，模型只能从合法 token 里采样。结果是输出 100% 合规，不需要后处理重试。

| 场景               | 用 tools                | 用 response_format       |
| ---------------- | --------------------- | ----------------------- |
| 需要让模型选择并触发外部副作用  | ✓                     | ✗                       |
| 只想拿一份合 schema 的数据 | 也能做但绕，要把"返回结果"写成假工具    | 直接                      |
| 客户端要不要执行回填        | 要                     | 不要                      |

各家的实现：

- OpenAI 的宽松模式：`response_format={"type": "json_object"}`，模型保证输出是合法 JSON，但不约束字段
- OpenAI 的严格模式：`response_format={"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}`，模型输出必须满足 schema，由解码器约束生成
- Anthropic 没有专门字段，常见技巧是 prefill。在 messages 末尾追加 `{"role": "assistant", "content": "{"}`，模型从 `{` 开始接续，配合 stop 序列控制结束
- Gemini 用顶层 `generationConfig.responseSchema` 参数
- 国产模型多数兼容 OpenAI 协议，`json_object` 普遍可用，`json_schema` 严格模式支持参差不齐，要看具体模型版本

实际 Agent 工程里两条路经常混用。需要副作用（查数据、跑代码、写文件）走 tools；需要纯结构化数据（分类标签、摘要、抽取字段）走 response_format。一个直观判断，需要执行什么动作就用 tools，只要拿一份合规数据就用 Structured Output。

### 2.4 返回结构，choices 完整解析

模型返回的结构如下，这是从 Moonshot 抓取的真实结果，这是第一轮对话之后，LLM API 返回的响应，返回的 message 的 role 是 assistant，而且明确要求调用本地的 tool 来获取当前的天气情况。

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "kimi-k2-0905-preview",
  "created": 1779288208,
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "get_weather:0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"深圳\"}"
            }
          }
        ],
        "reasoning_content": "用户询问深圳天气..."
      }
    }
  ],
  "usage": {"prompt_tokens": 65, "completion_tokens": 78, "total_tokens": 143}
}
```


完整的字段解析如下：

| 字段路径                              | 用途                      | Agent 必读 |
| --------------------------------- | ----------------------- | -------- |
| `choices[0].finish_reason`        | 终止原因，最权威的信号             | 必读       |
| `choices[0].message.content`      | 自然语言文本                  | 必读       |
| `choices[0].message.tool_calls`   | 工具调用清单（数组）              | 必读       |
| `tool_calls[].id`                 | 配对凭证                    | 必读       |
| `tool_calls[].function.name`      | 函数名                     | 必读       |
| `tool_calls[].function.arguments` | JSON 字符串，需 `json.loads` | 必读       |
| `message.reasoning_content`       | 思维链（厂商扩展）               | 否        |
| `usage.total_tokens`              | 成本                      | 否        |

### 2.5 finish_reason，Agent 调度的信号

finish_reason 表示这一轮 LLM 对话结束的原因。

正常情况下如果 LLM 当前需要调用 tool 来获取更多的信息，那么 Agent 就需要将 tool 调用的结果附加到 message 中，然后进行新一轮的交互。

如果 LLM 认为当前的交互已经完成，产出了最后的内容，那么 Agent 就可以直接将结果返回给用户，结束整个任务。

如果撞到了 max_tokens 上限或者触发了内容审查，那么整个任务也需要结束。

| 值                | 含义              | Agent 该做什么  |
| ---------------- | --------------- | ----------- |
| `tool_calls`     | 模型生成了工具调用 token | 执行工具，继续循环   |
| `stop`           | 模型自然说完（吐 EOS）   | 当最终答复，break |
| `length`         | 撞到 max_tokens   | 内容截断，要扩容    |
| `content_filter` | 安全审查拦截          | 报错或重试       |

代码里最简单的判断只看 tool_calls 和 content：

```python
if msg.tool_calls:        # 分支 A
    ...
elif msg.content:         # 分支 B
    break
```

为什么 `content` 非空就能当最终答复？因为协议设计上 `content` 和 `tool_calls` 语义互斥。

- 模型决定调工具，`tool_calls` 非空，`content` 通常为 `""` 或 `None`
- 模型决定最终答复，`content` 是文本，`tool_calls` 为 `None`

但实务中还有第三种边走边说的情况。`content` 是前言（如"我来查一下"），同时 `tool_calls` 也存在。`if/elif` 的顺序在这里关键。优先看 `tool_calls`，只有它为空时 `content` 才被当成终态。

更严谨的写法是结合 `finish_reason` 双重保险。

```python
if msg.tool_calls:
    ...                                  # 继续循环
elif resp.choices[0].finish_reason == "stop":
    break                                # 真正的终态
elif resp.choices[0].finish_reason == "length":
    raise RuntimeError("响应被截断")
```

### 2.6 ID 配对

这是工具调用回填时最容易踩的坑。多轮 tool 调用时，必须把对应的 id 严格配对：

```python
# 模型这次返回的 tool_call.id = "get_weather:0"
# 你必须把工具结果塞回去时，tool_call_id 等于这个 ID

messages.append({
    "role": "tool",
    "tool_call_id": "get_weather:0",    # ← 必须配对
    "name": "get_weather",
    "content": '{"temperature": "15C"}'
})
```

如果 ID 漏写或写错，下一轮请求会被服务端拒收，提示 `Tool call id 'xxx' not found in preceding assistant message`。

为什么需要 ID？因为模型可能一次返回多个 tool_calls（并行调用），用 name 区分不够。同一个工具可能被并行调好几次，比如对比深圳、上海、广州的天气。ID 是唯一标识。

### 2.7 上下文持续增长

在与 LLM 交互的过程中，可以发现 messages 数组的长度会越来越长，观察上面示例 Agent Loop 几轮后的 messages 状态。

```
轮1 请求前: 2 条 [system, user]

轮1 工具回填后: 4 条
  #0 [system]     你是助手...
  #1 [user]       深圳天气？
  #2 [assistant]  ↳ tool_call id=get_weather:0  get_weather(args={...})
                  "我来帮您查询..."
  #3 [tool]       name=get_weather  tool_call_id=get_weather:0
                  {"temperature": "15C", ...}

轮2 终态: 5 条
  ... 以上 4 条 ...
  #4 [assistant]  根据查询结果，深圳 15℃ 大风...
```

模型每次思考-行动的轨迹都被原样保留在上下文里，下一轮模型看着这些痕迹做综合。

- 如果你删掉 #2，模型会因为找不到对应的 tool_call 而报错。
- 如果你删掉 #3，模型会再次请求调用同一个工具，进入死循环。

每多调一轮工具，prompt 就变长，token 成本叠加。生产 Agent 通常会做 max_loops 兜底、上下文窗口管理、prompt 缓存。

## 3. LLM API 详解

第 2 节把 Chat Completions 这套协议的核心字段过了一遍。本节再补三件相关的事：function calling 协议自身的演进、流式响应、以及市面上另外两套主流接口（Anthropic Messages 与 OpenAI Responses）的差异。

### 3.1 function calling 的两代演进

Chat Completions 是事实标准，但它内部也经历过一次重要演进。OpenAI 的 function calling 协议有两代，理解差异能避免迁移老代码时踩坑。

| 维度              | 旧版 v1 (2023.06)             | 新版 v2 (2023.11+)                               |
| --------------- | --------------------------- | ---------------------------------------------- |
| 请求字段            | `functions=[...]`           | `tools=[{"type":"function","function":{...}}]` |
| 选择策略            | `function_call="auto"`      | `tool_choice="auto"`（多了 `"required"`）          |
| 响应里的工具调用        | `message.function_call`（对象） | `message.tool_calls`（数组）                       |
| 调用 ID           | 无                           | `tool_calls[i].id`                             |
| 工具类型字段          | 无                           | `tool_calls[i].type`（目前 `"function"`）          |
| 工具结果 role       | `function`                  | `tool`                                         |
| 配对字段            | 仅靠 `name`                   | `tool_call_id`                                 |
| `finish_reason` | `"function_call"`           | `"tool_calls"`                                 |
| 并行调用            | 不支持                         | 支持                                             |

新版有三个关键的改进：
1. 数组（并行调用）。一次返回多个 tool_calls。
2. id（强配对）。多调用追踪靠 ID 而非 name。
3. type（扩展位）。为 web_search、code_interpreter 等内置工具留口子。

很多国产模型基本只实现 v2，这也是目前 LLM 调用 tools 的标准用法。

### 3.2 流式响应

前面所有讨论都默认一次请求等到完整响应。但 Claude Code、Cursor、ChatGPT 这些产品里字一个个蹦出来的体验，是流式响应在跑。Agent 工程里几乎所有面向用户的场景都开流式。

打开方式很简单，加一个 `stream=True`。

```python
stream = client.chat.completions.create(
    model="kimi-k2-0905-preview",
    messages=messages,
    tools=tools,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

返回不再是一个完整 JSON，而是一串 Server-Sent Events。每个事件是一行 `data: {...}\n\n`，最后以 `data: [DONE]\n\n` 收尾。每个事件里 `choices[0].delta` 是这一片段的增量，而不是累计值。

**流式下读 content 容易，读 tool_calls 是个坑**

非流式下 `tool_calls` 是一个完整数组，参数 `arguments` 是一段 JSON 字符串。流式下整个 tool_calls 被切成多片。第一片包含 id、name 和空 arguments，后面每一片只带一小段 arguments 文本，要在客户端按 index 累加才能拼回完整 JSON。

```python
tool_buffers = {}            # index → {"id": ..., "name": ..., "arguments": ""}

for chunk in stream:
    delta = chunk.choices[0].delta
    for tc in (delta.tool_calls or []):
        buf = tool_buffers.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
        if tc.id:                  buf["id"] = tc.id
        if tc.function.name:       buf["name"] = tc.function.name
        if tc.function.arguments:  buf["arguments"] += tc.function.arguments

# 全部 chunk 处理完之后再 json.loads(buf["arguments"])
```

如果你在中间某一片就调 `json.loads`，多半会因为 JSON 还没传完而解析失败。

整轮结束的判断不能靠某一片的内容，要等到一个 chunk 的 `choices[0].finish_reason` 不是 None 才算到底。提前 break 会丢失最后几个 token。

streaming 的好处是用户随时可以叫停。HTTP 层面就是 close connection，OpenAI / Anthropic 都会尽快终止生成。但已经计费的 input token 不退，已经吐出的 output token 也照算。Agent 实现里的「停止」按钮通常就是触发这个动作。

### 3.3 其他主流 LLM API

Chat Completions 是事实标准，但当前 LLM 生态大体上也是按照类似的规则来组织 API，但是会有一些细微的差别。

**Anthropic Messages API

最关键的差异是 system 不在 messages 里，作为顶层独立参数。role 只有两个（user / assistant），且必须严格交替。工具调用和工具结果都打包进 content block，工具结果归在 user 这边。思维链是原生 content block。

```python
client.messages.create(
    model="claude-sonnet-4-5",
    system="...",                              # ← 顶层
    messages=[
        {"role": "user", "content": "深圳天气？"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"location": "深圳"}}
        ]},
        {"role": "user", "content": [          # ← 工具结果归在 user
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "..."}
        ]},
    ],
    tools=[...],
)
```

设计理念上 Anthropic 严格遵守对话只有双方的假设。system 是元数据，工具结果是给模型的外部输入，归类到 user。流式事件也比 Chat Completions 细，按 content block 分了 `message_start`、`content_block_start`、`content_block_delta`、`content_block_stop`、`message_stop` 几类。

**OpenAI Responses API**

2025 年 3 月 OpenAI 推出的新接口，目的是优化 Agent 场景与 LLM 的交互效率。最核心的改变是把状态搬到了服务端。客户端只发当前轮的 `input` 加一个 `previous_response_id` 指针，后端凭 ID 找到历史接续生成。同时内置了 `web_search`、`file_search`、`code_interpreter`、`computer_use` 等工具，整套 Agent Loop 在云端完成，外部看是一次调用。

```python
client.responses.create(
    model="gpt-4.1",
    instructions="...",                    # ← 类似 system
    input="...",
    tools=[
        {"type": "web_search_preview"},        # ← 服务端内置工具
        {"type": "function", "function": {...}},  # ← 自定义工具混用
    ],
    previous_response_id=last_resp.id,
    store=True,
)
```

Responses 把状态、内置工具、服务端 Agent Loop 都搬到了 OpenAI 那边，对原型期开发友好。生产场景下如果要自己掌控供应商和数据流，多数 Agent 框架还是回到 Chat Completions。

## 4. 现代 Agent 工具生态

Agent 在工程上落地不是像上面的结构那么简单，还需要考虑成本和效果，下面的这些是 Agent 能力扩展的关键，都是建立在 LLM API 之上的能力扩展。

### 4.1 Reasoning models 与推理预算

普通 LLM 拿到问题就直接吐答复 token，没有显式的思考阶段。Reasoning model 不一样，它会先输出一段思考 token，再基于这段思考生成最终答复。整个过程从「问题 → 答复」变成「问题 → 一段思考过程 → 答复」。这段思考叫 reasoning tokens 或 thinking tokens，对用户通常不可见，但模型自己会把它当作上下文继续往下推。

代价是延迟更高、token 成本更高（思考也要计费），收益是数学、代码、复杂规划这类任务上的准确率显著提升。简单的闲聊、分类、写作任务用推理模型反而更慢更贵，未必更好。

OpenAI 的 o1/o3、Anthropic 的 extended thinking、DeepSeek R1、Kimi K2、QwQ，都属于这一类。各家通过参数控制思考的深浅，调用方可以在快但浅和慢但深之间取舍。

各家的接口形态不一样。

```python
# OpenAI o1 / o3
client.chat.completions.create(
    model="o3-mini",
    messages=[...],
    reasoning_effort="medium",   # low / medium / high
)

# Anthropic extended thinking
client.messages.create(
    model="claude-sonnet-4-5",
    messages=[...],
    thinking={"type": "enabled", "budget_tokens": 10000},
)

# DeepSeek R1 / Kimi K2 等国产模型
# 多数走 reasoning_content 扩展字段，调用方式和普通 chat 一致
# 思考过程在 message.reasoning_content 返回
```

**计费上有个坑**。OpenAI 的 reasoning tokens 算在 `completion_tokens_details.reasoning_tokens` 里，按 output 价格收费，但默认不返回原文。也就是说你看不到模型想了什么，但要为这部分 token 付钱。

**Agent 里要不要把 reasoning 塞回 messages**。Anthropic 的 thinking block 在 multi-turn tool use 场景下要原样回填，否则模型会觉得自己上一轮的工具调用「凭空发生」。OpenAI 的 reasoning 默认不返回，问题不大。这里各家行为不一样。

### 4.2 Prompt caching

Agent loop 里 messages 越涨越长，每轮都重发整段 system + tools 描述特别浪费。Prompt caching 就是在服务端把这些常态化前缀缓存起来，命中时大幅降价。

OpenAI Chat Completions 是**自动**缓存。超过 1024 token 的前缀自动进缓存，命中后这部分按原价折扣计费。客户端不用做任何标记，但前缀必须**逐字一致**才能命中，所以 messages 顺序、system 内容、tools schema 要稳定。

Anthropic 是**显式**标记。在某条消息或 tools 数组里加 `cache_control={"type": "ephemeral"}`，把这条之前的所有内容打包进一个 cache breakpoint。最多 4 个 breakpoint。命中时缓存读取价很低，未命中写入价就要涨价。

```python
# Anthropic 显式缓存
client.messages.create(
    model="claude-sonnet-4-5",
    system=[
        {"type": "text", "text": "..." * 5000,
         "cache_control": {"type": "ephemeral"}},   # ← 这之前都缓存
    ],
    messages=[...],
)
```

Agent loop 里命中率最高的写法：把 **system prompt** 和 **tools schema** 放最前面（这两个最稳定），动态对话历史放后面。每轮请求只有末尾 messages 不一致，前缀可以全部命中。Claude Code 等这些这些产品的请求都是这么组织的。

TTL 短，OpenAI 5-10 分钟，Anthropic 默认 5 分钟（带付费选项可以延到 1 小时）。所以缓存对**短时间高频调用**最划算，对长间隔的对话作用有限。

### 4.3 工具的几种供应形式

在上面第二节有聊到 tools 字段是模型看到的工具菜单。但这份菜单怎么来？写死在 Agent 客户端代码里是最简单的方式，但这种硬编码会限制 Agent 的扩展能力。当下的 Agent 实践里，工具可以从四种不同的形态传递给 LLM。

**本地 function tool**

最基础的形态。Agent 客户端代码里定义一个 Python / TypeScript 函数，再写一份匹配的 JSON Schema 放到 tools 数组里。第 1 章那个 `get_weather` 例子就是这种。

```python
def get_weather(location): ...

tools = [{"type": "function", "function": {
    "name": "get_weather", "parameters": {...},
}}]
```

启动时注册，运行时模型决定调，客户端找函数执行回填。简单直接，缺点是每个 Agent 都得自己写一遍。

**CLI**

function tool 的一类特例。函数体里不是业务逻辑，而是 `subprocess` 调本地命令行。Claude Code 内置的 `Bash`、`Read`、`Write`、`Glob`、`Grep` 都是这种形态。

```python
def bash(command: str, timeout: int = 60):
    result = subprocess.run(command, shell=True, capture_output=True, timeout=timeout)
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}

tools = [{"type": "function", "function": {
    "name": "Bash", "parameters": {...},
}}]
```

模型看到的还是一个普通 function tool，但执行时落到本地 shell。这种形态让 Agent 直接获得整个命令行生态的能力，代价是要做沙箱、超时、权限控制。

**MCP（Model Context Protocol）**

2024 年底 Anthropic 推的开源协议，把工具供应的进程独立出去。Claude Desktop、Claude Code、Cursor、Cline、Codex CLI 现在都支持接 MCP server。

工作分工：

- **MCP server**：提供工具的独立进程。可以是 GitHub、Figma、Postgres、本地文件系统、任何外部能力的 wrapper。一个 server 暴露若干工具
- **MCP host**：Agent 客户端。启动时扫描配置好的 MCP server 列表，把每个 server 暴露的工具收集起来，转换成 LLM API 的 tools schema 格式喂给模型

```
[MCP server: github]      [MCP server: postgres]
        ↓                          ↓
        └────── stdio/HTTP ────────┘
                       ↓
              [MCP host: Claude Code]
                       ↓
            转换为 tools=[...] schema
                       ↓
                  [LLM API]
```

对模型来说毫无差别，看到的还是一个标准的 tools 数组。但对 Agent 生态意义巨大，工具不再绑死在某个客户端代码里。给 Claude Code 写一个 `github-mcp-server`，其他的 Agent 也能直接复用。

**Skills**

Skills 不是工具，是**操作手册**。这点需要注意一下，因为它经常被和 MCP 混淆。

Anthropic 对 skills 的设计大致是下面这种结构：
```
.claude/skills/
├── pdf-form-filling/
│   ├── SKILL.md          ← 触发条件 + 操作步骤
│   └── templates/        ← 资源文件
├── excel-analysis/
│   └── SKILL.md
└── tweet-writer/
    └── SKILL.md
```

每个 Skill 包含一段 描述触发条件（比如「用户提到推文、整理成推文时使用」），加上主体说明这类任务怎么做、用什么工具、有哪些规则。

模型在对话中识别到任务类型符合某个 skill 的触发条件时，把对应的 SKILL.md 注入到 context（通常是 system prompt 或独立 message）。注入的是做事方法，不是能力。能力（tools）已经通过 tools 数组或 MCP 提供了，Skills 只是告诉模型怎么用好这些能力解决某类任务。

Skills 的关键特性是**按需加载**。不是把所有 skill 内容一股脑塞进 system prompt，而是任务相关时才注入对应的那一份。这样既能积累大量领域知识，又不会撑爆上下文。


**四者的关系**

四种形态在层级上完全不同，通常不会产生冲突，而且在很多场景下会搭配使用。

| 形态            | 解决什么                       | 形态                     |
| ------------- | -------------------------- | ---------------------- |
| function tool | 模型能调什么具体能力                 | tools 数组里一项            |
| CLI 包装        | function tool 的特例，让模型用本地命令 | tools 数组里一项，函数体调 shell |
| MCP server    | 工具供应方跨客户端复用                | 独立进程 + 标准协议            |
| Skills        | 模型遇到这类任务该怎么做               | system / context 注入    |

一个 MCP server 可以暴露 CLI 包装的工具（比如 `github-mcp-server` 内部就是调 `gh` CLI）。一个 Skill 的 SKILL.md 可以引用本地 function tool 或 MCP 提供的工具。同一个 Agent 同时挂内置 CLI 工具加几个 MCP server 加一堆 Skills 是常态。

## Take Away

到这里，你应该能回答最开始那个问题。Agent 是怎么工作的？

模型从来不做任何事，它只输出 token。while 循环把模型输出的 tool_calls JSON 翻译成函数调用，把执行结果翻译回 messages 塞回模型，让模型再决策一次。这个来回翻译的过程，就是 Agent。

ChatGPT、Claude Code、Codex 这些看上去会自己动手的产品，底子是同一段 while 循环。只是工具更多（Bash、Read、Write、Glob 等），模型更强（GPT-5-Codex、Claude Sonnet），协议更精细（Responses、Messages）。理解了上面这套机制，你看任何 Agent 框架，看到的都是同一件事。
<img width="715" height="549" alt="Clipboard_Screenshot_1779720251" src="https://github.com/user-attachments/assets/18c0a38d-f868-4f22-a3c6-4705a2a2d35a" />

## Ref

[1] https://www.anthropic.com/engineering/building-effective-agents
[2] https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_role.py
