# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36

#读取数据集，jsonl格式（json lines），每行是一个json对象，包含conversations字段，里面是对话内容，一行就是一个完整的 JSON 对象，多行组成一个数据集。
def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f): #enumerate的作用是在遍历可迭代对象时，同时获取元素的索引和值。它返回一个迭代器，每次迭代返回一个包含两个元素的元组，第一个元素是索引，第二个元素是对应的值。
            #line 末尾会自带换行符
            if i >= 10000: break # 选10000行测试
            try:  #try ... except ... 的目的就是：某一行坏了不要让整个程序中断，跳过坏行继续处理下一行。
                data = json.loads(line) #就是从字符串中加载 JSON,把json格式转化为Python对象, line:{"conversations":[{"role":"user","content":"什么是 tokenizer？"},{"role":"assistant","content":"tokenizer 是把文本切分成 token 的工具。"}]}
                #转化为python对象后，data是一个字典，里面有一个键 'conversations'，对应的值是一个列表，列表中每个元素都是一个字典，包含 'role' 和 'content' 两个键。
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                """
                #data数据类型是字典,上述是一个列表推导式,用于从 data 中提取所有对话内容(content)并存储在 contents 列表中。它会遍历 data 中的 conversations 列表,对于每个对话项(item),如果该项包含 content 字段且不为空，就将其添加到 contents 列表中。
                #字典的 .get(key, default) 方法表示：尝试获取字典中指定 key 的值，如果 key 不存在，则返回默认值 default。这里的 default 是空列表 []，表示如果 conversations 不存在，就返回一个空列表。
                item的数据类型是字典,item.get('content')尝试获取当前对话项的 content 字段的值，如果不存在则返回 None。if item.get('content')是一个条件判断，只有当 content 字段存在且不为空时，才会将其加入 contents 列表。
                for item in data.get('conversations', [])遍历对话列表中的每一项。
                item.get('content')从当前这轮对话里取 content 字段。
                if item.get('content')这是过滤条件。只有当 item.get('content') 有值时，才把它加入列表。
                """
                if contents:
                    yield "\n".join(contents)  #注意其返回的是字符串，不是列表
                """
                join 是字符串方法，用来把一个字符串列表拼接成一个大字符串。/n是换行符, yield "\n".join(contents)意思是：把 contents 列表里的所有对话内容用换行符连接成一个字符串，并通过 yield 产出这个字符串。
                yield 和 return 有点像，都能从函数里“产出”结果。 但区别是,return:一次性返回一个结果.然后函数结束。yield:每次产出一个结果,函数暂停;下次需要数据时,从暂停处继续运行。
                包含 yield 的函数叫做生成器函数。
                所以texts = get_texts(data_path)并不会马上读取完整文件，而是返回一个生成器对象。 后面 tokenizer 训练器会一条一条取:tokenizer.train_from_iterator(texts, trainer=trainer)，节省内存。
                
                """
            except json.JSONDecodeError:
                continue
#训练tokenizer分词器，分词器的训练过程包括以下几个步骤：
#1. 初始化一个空的分词器对象。
#2. 设置预分词器（pre-tokenizer），用于将输入文本拆分为更小的单元（如单词或子词）。
#3. 定义特殊的token列表，这些token在训练过程中会被保留，并且不会被分词器拆分。
#4. 创建一个训练器对象，指定词汇表大小、特殊token列表等参数。
#5. 从数据集中获取文本，并使用训练器对分词器进行训练。
#6. 设置解码器（decoder），用于将分词后的token序列转换回原始文本。
#7. 将训练好的分词器保存到指定目录  
def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    #ByteLevel 预分词包括处理空格-> 使用规则/正则切成初步片段-> 转成 byte-level 可训练表示
    #Tokenizer是一个类，用于创建和管理分词器对象。它是Hugging Face Tokenizers库中的核心类，提供了分词、解码、保存和加载等功能。
    #models是子模块，一个 .py 文件通常就是一个模块，可以import，把一组相关的变量、函数、类放在一起。BPE是一个类， models.BPE() 返回的是一个 BPE mode
    # 实例，tokenizer是一个对象， tokenizer.pre_tokenizer是 tokenizer 对象的一个属性。pre_tokenizers是一个模块，ByteLevel是一个类，
    # pre_tokenizers.ByteLevel(add_prefix_space=False)返回一个 ByteLevel pre-tokenizer 实例，并将其赋值给 tokenizer.pre_tokenizer 属性。
    
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    #真正的特殊控制 token，用于对话边界、多模态占位、BOS/EOS/PAD 等。
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    # 额外加入词表的结构标记，用于工具调用、工具响应、思考标签等，但后面刻意不标记为 special=True。解码decoder时，仍然会把这些 token 当成普通 token 处理，不会被特殊处理,不会被跳过，容易保留。
    # 额外加入词表，但后面不一定被当成特殊token使用，
    """
    特殊 token 不靠普通语料训练出来；但需要在 tokenizer 训练阶段提前注册进词表。训练 tokenizer 的普通数据可以不包含特殊 token,
    后续构造模型训练样本或聊天输入时,chat_template 才会插入特殊 token。提前注册的目的,是保证这些 token 在编码时是整体 token,而不是被拆碎。
    分配id:先有：
      special tokens
      256 个 ByteLevel 基础字符
    然后统计语料中的相邻 token pair


  """
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)#计算还需要预留多少个 buffer token
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 用列表推导式生成 buffer token,预留一定数量的token位置,方便未来扩展
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),#表示：训练 BPE 时，初始词表里应该先包含哪些基础字符/符号。返回所有 256 个可能的字节值，不含id
        #   - 确保所有单字节都被包含在初始词汇表中
        #   - 这样可以避免字节级 OOV 问题，任何字节组合都能被编码
        #alphabet() 方法(是一个函数）它返回一个列表，里面是 ByteLevel tokenizer 使用的一批基础符号/字符。返回一个包含所有可能的单字节字符的集合，确保在训练 BPE 模型时，初始词汇表中包含所有单字节字符，从而避免在编码过程中出现未登录的字节（OOV）问题。
        special_tokens=all_special_tokens
    )#创建训练器，训练器设置，定义“训练 tokenizer 时应该遵守哪些规则”， BpeTrainer 是里面的一个类，专门用来训练 BPE 分词模型。
    texts = get_texts(data_path)#获取训练数据
    tokenizer.train_from_iterator(texts, trainer=trainer)#开始训练，train_from_iterator 方法会从迭代器中获取文本数据，并使用指定的训练器对分词器进行训练。训练过程中，分词器会根据提供的文本数据学习如何将文本拆分为 token，并构建词汇表。
    #训练完成后，tokenizer 对象内部就有了训练好的 BPE 词表和合并规则。
    tokenizer.decoder = decoders.ByteLevel()#设置解码器
    #decoder 是 tokenizer 的一个属性，用于将 token 序列转换回原始文本。这里设置为 ByteLevel decoder，表示解码时会按照字节级别进行处理，确保能够正确还原原始文本。
    tokenizer.add_special_tokens(special_tokens_list)
    #不只是“加入词表”，而是把 special_tokens_list 里的 token 标记成真正的 special token。
    os.makedirs(tokenizer_dir, exist_ok=True)#创建保存目录
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))#保存完整 tokenizer配置 到 tokenizer.json。
    """模型类型、词表、BPE merges、pre_tokenizer 配置decoder 配置、added_tokens、special token 信息
  后续可以用它加载 tokenizer。"""
    tokenizer.model.save(tokenizer_dir)#保存 BPE 模型文件到指定目录，包含 vocab.json 和 merges.txt。
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:#用utf-8编码读取文件中的内容，返回一个文件对象 f
        tokenizer_data = json.load(f)#从文件读取json转化为python字典
    for token_info in tokenizer_data.get('added_tokens', []):#added_tokens是键，值是列表，存放的是额外添加的token信息，每个元素是一个字典，包含 token 的内容、是否是特殊 token 等信息。
        if token_info['content'] not in special_tokens_list: #token_info 是字典，token_info['content']是token的内容，判断是否在 special_tokens_list 中
            token_info['special'] = False#确保只有真正的特殊词元才是true，其他的都标记为 False,比如additional_tokens_list，buffer_tokens
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:#写入，覆盖原文件的内容
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)#中文不转义，缩进2个空格
    #json.dump()的作用是什么：将Python对象转换为JSON格式并写入文件
    #那如何直接将Python对象转换为JSON格式字符串呢？可以使用json.dumps()方法，它会返回一个字符串，而不是写入文件。
    #那如何直接将python对象写入文件呢？

    added_tokens_decoder = {}#创建一个空字典，把后面会把 token id 到 token 信息的映射放进去。
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)#查询某个 token 对应的 token id。
        added_tokens_decoder[str(idx)] = { #将数字转化为字符串，因为 JSON 对象的 key 本质上是字符串。后面写进 tokenizer_config.json 时，用字符串更符合 JSON 格式。
            "content": token,
            "lstrip": False,#True，可能会处理 token 左侧空格。
            "normalized": False,#不参与 normalizer 的规范化处理。
            "rstrip": False,
            "single_word": False,#表示这个 token 不要求必须是一个独立单词。
            "special": True if token in special_tokens_list else False
        }

    config = {
        "add_bos_token": False, #是否自动在输入开头添加 BOS（Begin of Sequence）token
        #   设置为 False，因为我们使用 <|im_start|> 手动标记消息开始
        "add_eos_token": False,
        "add_prefix_space": False,#是否在输入文本前添加空格，通常用于处理以空格开头的文本
        "added_tokens_decoder": added_tokens_decoder,#特殊 Token 的解码映射
        #   键是 token 的 ID（字符串格式），值包含 token 的详细属性
        #   这个映射告诉 Transformers 如何解码这些特殊 token
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],#通常放的是那些除了标准特殊 token 之外的额外特殊 token。，但<|endoftext|>映射为pad_token，所以不放在这里
        "bos_token": "<|im_start|>",#序列开始符，映射到 <|im_start|>，用于标记对话消息的开始
        "clean_up_tokenization_spaces": False,#是否清理分词后的空格
        "eos_token": "<|im_end|>",#属于标准特殊 token，序列结束符，映射到 <|im_end|>，用于标记对话消息的结束，可直接使用，tokenizer.bos_token
        "legacy": True,#是否使用旧版兼容模式
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},# SentencePiece 模型参数（BPE 不使用，留空）
        "spaces_between_special_tokens": False,#特殊token之间是否保留空格
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
        #作用是告诉 Hugging Face transformers：加载这个 tokenizer 时，应该用 PreTrainedTokenizerFast 这个 tokenizer 类。
        #应为这是我们自己训练的分词器，不像已经注册好的模型 LlamaTokenizerFast，PreTrainedTokenizerFast，是 Hugging Face transformers 里的一个通用 fast tokenizer 包装类，它会利用之前保存的tokenizer.json
        #聊天消息格式化模板（Jinja2 语法），这是本分词器最核心的配置之一，定义了如何把多轮对话 messages 转换成模型真正看到的一段 prompt 文本,统一聊天格式，尤其是 SFT 训练和推理时，如果格式不一致，模型会学乱。
    }
    """
         messages = [
      {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
      {"role": "user", "content": "你来自哪里？"},
      {"role": "assistant", "content": "我来自月球"}]

        转换成类似：<|im_start|>system
                你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
                <|im_start|>user
                你来自哪里？<|im_end|>
                <|im_start|>assistant
                <think>

                </think>

                我来自月球<|im_end|>

             也就是说,chat_template 定义的是聊天数据的序列化格式。
        """
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")
#保存配置文件到 tokenizer_config.json，这个文件是 Transformers 库加载分词器时必需的配置文件，Transformers 库的兼容配置文件

#用tokenizers这个库可以训练自己的分词器，那怎么使用训练好的分词器呢，就需要用另一个库transformers，通过AutoTokenizer.from_pretrained加载tokenizer_config.json和tokenizer.json这两个文件
# 得到一个tokenizer对象（PreTrainedTokenizerFast）带有词表和词元合并的信息。然后就可以对输入数据集进行处理了。
# AutoTokenizer.from_pretrained不会读取vocab.json和merges.txt这两个文件，属于额外保存的信息，应为json已经自带了。


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)#autotokenizer是一个类， from_pretrained是一个方法，返回一个分词器对象。返回具体的 tokenizer 类实例，（PreTrainedTokenizerFast）取决于 tokenizer_config.json 中的 "tokenizer_class" 配置。
    #会根据 tokenizer_config.json 自动选择合适的 tokenizer 类，加载 tokenizer.json 里的词表和配置。
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False#只应用 chat template，把 messages 转成字符串，不要立刻分词成 token ids。
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))# __len__魔法函数，其函数内部定义了返回的内容。 return self.vocab_size + len(self.added_tokens_encoder)，
    #added_tokens_encoder这种是在 tokenizer 已经训练好、已经加载之后，再额外加 token。
    model_inputs = tokenizer(new_prompt)#返回的 model_inputs 通常是一个类似字典的对象，里面有："input_ids": [...],"attention_mask": [...]
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-'*100)
    print('压缩率测试（Chars/Tokens）：')#字符数 / token数，平均一个token能表示多少个字符，越大越好，说明tokenizer压缩能力强。
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:#有些字符可能需要多个 token 才能组成完整可解码字符串，避免出现乱码
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:#current_decode需非空，且不包含 '\ufffd'，'\ufffd'是 Unicode 的替代字符，通常表示无法解码的字节或字符。也就是说，只有当当前缓冲区的 token 可以被成功解码为有效的字符串时，才会打印输出。
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache#决定打印时 token id 怎么展示
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
