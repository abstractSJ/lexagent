"""
本地私密配置模板。

复制本文件为 agent_system/config_local.py（该文件已被 .gitignore 排除），
填入真实 API Key 后即可本地运行；也可以不建此文件，改用环境变量注入。
读取优先级：环境变量 > config_local.py > config.py 内置默认值。
"""

LOCAL_ENV_DEFAULTS = {
    "AGENT_LLM_API_KEY": "请填入你的 LLM API Key",
    "AGENT_LLM_BASE_URL": "请填入 OpenAI-compatible 服务地址",
    "AGENT_LLM_MODEL": "gpt-5.5",
    "BOCHA_WEB_SEARCH_API_KEY": "请填入你的博查 Web Search API Key",
}
