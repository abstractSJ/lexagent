"""
OpenAI Responses API 客户端封装。

这个模块负责封装“单次 LLM 请求”的底层细节，例如 OpenAI-compatible SDK 调用、
流式输出、图片转 base64 data URL、usage 读取等。

对外仍然接收项目内部的 Chat-style messages，原因是 ChatSession 和当前测试脚本已经
基于 system/user/assistant 这种简单消息格式工作；对内会在发请求前转换为 Responses API
需要的 input / instructions 结构。这样既能保留学习项目已有调用方式，也能为后续
任务执行型 Agent 的工具调用、事件流和推理配置打基础。
"""

from pathlib import Path
import base64
import mimetypes
from typing import Any, Dict, Iterator, List, Sequence

from openai import OpenAI

from agent_system.config import LLMCallOptions, LLMConfig


# Message 表示项目内部的一条 Chat-style 消息，例如：{"role": "user", "content": "你好"}。
# content 也可以是多模态列表，例如文本块 + 图片块；发送前会被转换为 Responses API input。
Message = Dict[str, Any]
ImagePath = str | Path


class OpenAIChatClient:
    """
    OpenAI-compatible LLM 客户端。

    这个类主要负责单次请求，不直接保存多轮聊天历史。
    纯文本和图片输入都通过同一套参数进入，区别只是是否传入 image_paths。

    类名保留 OpenAIChatClient 是为了兼容当前项目已有导入；实际请求已经切换到
    Responses API，这样后续更容易接入工具调用和任务执行型 Agent 流程。

    Args:
        config: LLM 配置对象，包含 api_key、base_url、model、temperature、reasoning_effort 等参数。
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        初始化 OpenAI-compatible 客户端。

        Args:
            config: LLM 配置对象。
        """

        self.config = config

        client_kwargs = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }

        # base_url 用于接入 OpenAI-compatible 服务；如果使用官方 OpenAI，可以不配置。
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> Iterator[str]:
        """
        使用 Responses API 流式调用模型，逐段返回文本。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选的本地图片路径。传入后会自动追加到最后一条 user message。
                可以传单张图片路径，也可以传多张图片路径列表。
            image_detail: 图片理解精度。常见值是 auto、low、high；具体支持情况取决于模型服务。
            options: 可选单次调用覆盖参数。为 None 时沿用当前客户端配置。

        Yields:
            str: 模型生成过程中的文本片段。

        Raises:
            ValueError: messages 为空、格式不正确或图片路径不合法时抛出。
            RuntimeError: 接口调用失败时抛出。
        """

        instructions, response_input = self._build_responses_request(
            messages=messages,
            image_paths=image_paths,
            image_detail=image_detail,
        )

        request_params: Dict[str, Any] = {
            **self._build_generation_params(options),
            "input": response_input,
            "stream": True,
        }

        # instructions 只在确实存在 system prompt 时传递。
        # 原因是部分 OpenAI-compatible 服务对显式 None 的兼容性不如官方 SDK。
        if instructions:
            request_params["instructions"] = instructions

        try:
            stream = self.client.responses.create(**request_params)
        except Exception as error:
            # 封装层统一抛 RuntimeError，避免上层 Agent 直接依赖 OpenAI SDK 的异常细节。
            raise RuntimeError(f"LLM Responses 流式调用失败：{error}") from error

        for event in stream:
            # Responses API 的流式返回是事件流，不再是 Chat Completions 的 choices.delta。
            # 这里只向上层暴露纯文本片段，保持 ChatSession 和 CLI 的使用方式稳定。
            text_delta = self._extract_text_delta(event)
            if text_delta:
                yield text_delta

    def complete(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        使用流式接口调用模型并返回完整文本。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选的本地图片路径。传入后会自动追加到最后一条 user message。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前客户端配置。

        Returns:
            str: 模型完整回复。
        """

        # complete() 继续保留原有“流式拼接”语义。
        # 原因是 planner 等现有代码已经依赖这个稳定入口；真正非流式测试走 complete_non_stream()。
        return "".join(
            self.chat(
                messages,
                image_paths=image_paths,
                image_detail=image_detail,
                options=options,
            )
        )

    def complete_non_stream(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        使用 Responses API 非流式调用模型并返回完整文本。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选的本地图片路径。传入后会自动追加到最后一条 user message。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前客户端配置。

        Returns:
            str: 模型完整回复。

        Raises:
            RuntimeError: 接口调用失败时抛出。
        """

        instructions, response_input = self._build_responses_request(
            messages=messages,
            image_paths=image_paths,
            image_detail=image_detail,
        )
        request_params: Dict[str, Any] = {
            **self._build_generation_params(options),
            "input": response_input,
        }
        if instructions:
            request_params["instructions"] = instructions

        try:
            response = self.client.responses.create(**request_params)
        except Exception as error:
            raise RuntimeError(f"LLM Responses 非流式完整调用失败：{error}") from error

        return self._extract_response_text(response)

    def get_usage(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> Dict[str, Any]:
        """
        发起一次 Responses API 请求并返回服务端 usage 信息。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选的本地图片路径。传入后会自动追加到最后一条 user message。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前客户端配置。

        Returns:
            Dict[str, Any]: 服务端返回的 usage 字段。如果服务端没有返回 usage，则返回空字典。
                为了兼容旧测试脚本，会把 input_tokens/output_tokens 映射为
                prompt_tokens/completion_tokens。

        Raises:
            RuntimeError: usage 请求失败时抛出。
        """

        instructions, response_input = self._build_responses_request(
            messages=messages,
            image_paths=image_paths,
            image_detail=image_detail,
        )

        request_params: Dict[str, Any] = {
            **self._build_generation_params(options),
            "input": response_input,
        }

        if instructions:
            request_params["instructions"] = instructions

        try:
            response = self.client.responses.create(**request_params)
        except Exception as error:
            # usage 必须通过一次真实请求获得，因为图片 token 往往由服务端按视觉输入重新计算。
            raise RuntimeError(f"LLM Responses usage 请求失败：{error}") from error

        return self._normalize_usage(getattr(response, "usage", None))

    def image_to_data_url(self, image_path: ImagePath) -> str:
        """
        把本地图片转换为 base64 data URL。

        Args:
            image_path: 本地图片路径。

        Returns:
            str: data URL，例如 data:image/jpeg;base64,...。
        """

        # 这个公开方法主要用于教学和调试，例如观察图片最终会以什么形式进入请求。
        # 正常业务调用时不需要手动调用它，直接把 image_paths 传给 chat()/complete()/get_usage 即可。
        return self._image_to_data_url(image_path)

    def build_responses_request(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
    ) -> tuple[str | None, List[Dict[str, Any]]]:
        """
        把项目内部 messages 转换为 Responses API 请求结构。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选图片路径。需要模型看图时传入。
            image_detail: 图片理解精度。常见值是 auto、low、high。

        Returns:
            tuple[str | None, List[Dict[str, Any]]]: instructions 与 input。
        """

        # AgentRunner 需要复用同一套消息转换逻辑。
        # 这里提供公开包装，避免上层直接依赖下划线私有方法。
        return self._build_responses_request(
            messages=messages,
            image_paths=image_paths,
            image_detail=image_detail,
        )

    def create_response(
        self,
        *,
        input: List[Dict[str, Any]],
        instructions: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        options: LLMCallOptions | None = None,
    ) -> Any:
        """
        发起一次非流式 Responses API 请求。

        Args:
            input: Responses API 的 input 列表。
            instructions: 可选系统指令。
            tools: 可选工具定义列表。
            previous_response_id: 可选上一轮 response id，用于工具调用后的续接。
            options: 可选单次调用覆盖参数。为 None 时沿用当前客户端配置。

        Returns:
            Any: OpenAI SDK 返回的 response 对象。

        Raises:
            RuntimeError: 接口调用失败时抛出。
        """

        request_params: Dict[str, Any] = {
            **self._build_generation_params(options),
            "input": input,
        }

        # 只传存在的可选字段，原因是兼容服务通常比官方接口更容易被 None 参数影响。
        if instructions:
            request_params["instructions"] = instructions
        if tools is not None:
            request_params["tools"] = tools
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id

        try:
            return self.client.responses.create(**request_params)
        except Exception as error:
            raise RuntimeError(f"LLM Responses 非流式调用失败：{error}") from error

    def _build_responses_request(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None,
        image_detail: str,
    ) -> tuple[str | None, List[Dict[str, Any]]]:
        """
        构造 Responses API 需要的 instructions 和 input。

        Args:
            messages: 项目内部 Chat-style 消息列表。
            image_paths: 可选图片路径。
            image_detail: 图片理解精度。

        Returns:
            tuple[str | None, List[Dict[str, Any]]]: system 指令文本和 Responses input 列表。

        Raises:
            ValueError: messages 无法转换为 Responses API 请求时抛出。
        """

        request_messages = self._build_request_messages(
            messages=messages,
            image_paths=image_paths,
            image_detail=image_detail,
        )
        self._validate_messages(request_messages)

        instruction_parts: List[str] = []
        response_input: List[Dict[str, Any]] = []

        for message in request_messages:
            role = message.get("role")
            if role == "system":
                instruction_text = self._system_content_to_text(message.get("content"))
                if instruction_text:
                    instruction_parts.append(instruction_text)
                continue

            response_input.append(self._message_to_responses_input(message))

        if not response_input:
            raise ValueError("messages 中除 system 外，必须至少包含一条 user 或 assistant 消息。")

        instructions = "\n\n".join(instruction_parts) if instruction_parts else None
        return instructions, response_input

    def _message_to_responses_input(self, message: Message) -> Dict[str, Any]:
        """
        将单条项目内部 message 转换为 Responses input item。

        Args:
            message: 项目内部 Chat-style 消息。

        Returns:
            Dict[str, Any]: Responses API 可接收的 input item。

        Raises:
            ValueError: 当前消息类型暂不支持转换时抛出。
        """

        role = message.get("role")
        content = message.get("content")

        if role == "tool":
            raise ValueError(
                "当前项目还没有接入 Responses 工具调用循环，暂不支持 tool message 转换。"
            )

        if role not in {"user", "assistant"}:
            raise ValueError(f"Responses input 暂不支持 role：{role}。")

        return {
            "role": role,
            "content": self._content_to_responses_blocks(role=role, content=content),
        }

    def _content_to_responses_blocks(
        self,
        role: str,
        content: Any,
    ) -> List[Dict[str, Any]]:
        """
        将项目内部 content 转换为 Responses API 内容块。

        Args:
            role: 当前消息角色。
            content: 原始 message content。

        Returns:
            List[Dict[str, Any]]: Responses API 内容块列表。

        Raises:
            ValueError: 内容块不支持转换时抛出。
        """

        text_block_type = "output_text" if role == "assistant" else "input_text"

        if content is None:
            # Responses API 的 message content 不适合传空列表。
            # 这里用空文本占位，原因是历史消息里偶尔可能出现 content=None，显式空文本更容易调试。
            return [{"type": text_block_type, "text": ""}]

        if isinstance(content, str):
            return [{"type": text_block_type, "text": content}]

        if not isinstance(content, list):
            raise ValueError("message content 必须是字符串、列表或 None。")

        response_blocks: List[Dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")

            if block_type in {"text", "input_text", "output_text"}:
                response_blocks.append(
                    {
                        "type": text_block_type,
                        "text": str(block.get("text", "")),
                    }
                )
                continue

            if block_type in {"image_url", "input_image"}:
                if role != "user":
                    raise ValueError("Responses API 中，图片输入只能放在 user message 中。")

                response_blocks.append(self._image_block_to_responses(block))
                continue

            raise ValueError(f"暂不支持转换 content block 类型：{block_type}。")

        if not response_blocks:
            raise ValueError("message content 转换后不能为空。")

        return response_blocks

    def _image_block_to_responses(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        将图片内容块转换为 Responses API 的 input_image 块。

        Args:
            block: Chat-style 或 Responses-style 图片块。

        Returns:
            Dict[str, Any]: Responses API input_image 内容块。

        Raises:
            ValueError: 图片块缺少 URL 时抛出。
        """

        block_type = block.get("type")
        detail = block.get("detail", "auto")
        image_url: str | None = None

        if block_type == "image_url":
            image_payload = block.get("image_url")
            if isinstance(image_payload, dict):
                image_url = image_payload.get("url")
                detail = image_payload.get("detail", detail)
            elif isinstance(image_payload, str):
                image_url = image_payload
        elif block_type == "input_image":
            image_url = block.get("image_url")

        if not image_url:
            raise ValueError("图片内容块缺少 image_url。")

        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": detail,
        }

    def _system_content_to_text(self, content: Any) -> str:
        """
        将 system message 的 content 转换为 Responses instructions 文本。

        Args:
            content: system message 的原始 content。

        Returns:
            str: 可传给 instructions 的纯文本。

        Raises:
            ValueError: system content 包含非文本内容时抛出。
        """

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            raise ValueError("system message 的 content 必须是字符串、文本块列表或 None。")

        text_parts: List[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type not in {"text", "input_text", "output_text"}:
                raise ValueError("system message 只能包含文本内容块。")
            text_parts.append(str(block.get("text", "")))

        return "\n".join(text_parts)

    def _extract_response_text(self, response: Any) -> str:
        """
        从非流式 Responses response 中提取最终文本。

        Args:
            response: OpenAI SDK response 对象或兼容 dict。

        Returns:
            str: 最终文本；没有文本时返回空字符串。
        """

        output_text = self._get_attr(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        text_parts: List[str] = []
        output_items = self._get_attr(response, "output", []) or []
        for item in output_items:
            if self._get_attr(item, "type") != "message":
                continue
            content_items = self._get_attr(item, "content", []) or []
            for content_item in content_items:
                if self._get_attr(content_item, "type") != "output_text":
                    continue
                text = self._get_attr(content_item, "text", "")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)

    def _get_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        """
        兼容 dict 与 SDK 对象的字段读取。

        Args:
            obj: dict 或 SDK 对象。
            name: 字段名。
            default: 默认值。

        Returns:
            Any: 字段值。
        """

        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _extract_text_delta(self, event: Any) -> str | None:
        """
        从 Responses API 流式事件中提取文本增量。

        Args:
            event: OpenAI SDK 返回的流式事件对象，或兼容服务返回的 dict 事件。

        Returns:
            str | None: 当前事件中的文本增量；非文本事件返回 None。
        """

        if isinstance(event, dict):
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                return delta if isinstance(delta, str) else None
            return None

        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            return delta if isinstance(delta, str) else None

        return None

    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:
        """
        归一化 Responses API 的 usage 字段。

        Args:
            usage: SDK 返回的 usage 对象、dict 或 None。

        Returns:
            Dict[str, Any]: 归一化后的 usage。会保留原始字段，并补充旧脚本使用的别名。
        """

        if usage is None:
            return {}

        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = dict(usage)
        else:
            # 有些兼容 SDK 对象不是 Pydantic model，也不是 dict。
            # vars() 可以尽量保留其公开属性，避免 usage 解析因为对象类型差异而失败。
            data = dict(vars(usage))

        if "prompt_tokens" not in data and "input_tokens" in data:
            data["prompt_tokens"] = data["input_tokens"]

        if "completion_tokens" not in data and "output_tokens" in data:
            data["completion_tokens"] = data["output_tokens"]

        if "total_tokens" not in data:
            input_tokens = data.get("input_tokens", data.get("prompt_tokens"))
            output_tokens = data.get("output_tokens", data.get("completion_tokens"))
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                data["total_tokens"] = input_tokens + output_tokens

        return data

    def _build_request_messages(
        self,
        messages: List[Message],
        image_paths: ImagePath | Sequence[ImagePath] | None,
        image_detail: str,
    ) -> List[Message]:
        """
        构造追加图片后的项目内部 messages。

        Args:
            messages: 原始消息列表。
            image_paths: 可选图片路径。
            image_detail: 图片理解精度。

        Returns:
            List[Message]: 已经追加图片内容的消息列表。
        """

        # 复制每条 message，避免把 base64 图片内容直接写回调用方维护的历史上下文。
        # 这样做可以防止 main.py 这类多轮聊天程序的 messages 被巨大 base64 字符串污染。
        request_messages = [dict(message) for message in messages]

        if image_paths is None:
            return request_messages

        normalized_paths = self._normalize_image_paths(image_paths)
        if not normalized_paths:
            return request_messages

        target_message = self._find_last_user_message(request_messages)
        original_content = target_message.get("content")
        content_blocks = self._content_to_blocks(original_content)

        for image_path in normalized_paths:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_data_url(image_path),
                        "detail": image_detail,
                    },
                }
            )

        target_message["content"] = content_blocks
        return request_messages

    def _content_to_blocks(self, content: Any) -> List[Dict[str, Any]]:
        """
        把原始 content 转成多模态 content block 列表。

        Args:
            content: 原始 message content。

        Returns:
            List[Dict[str, Any]]: 多模态内容块列表。

        Raises:
            ValueError: content 类型不支持时抛出。
        """

        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if isinstance(content, list):
            # 复制列表本身，避免 append 图片块时修改调用方传入的原始列表。
            return list(content)

        raise ValueError("追加图片时，最后一条 user message 的 content 必须是字符串、列表或 None。")

    def _find_last_user_message(self, messages: List[Message]) -> Message:
        """
        查找最后一条 user message。

        Args:
            messages: 消息列表。

        Returns:
            Message: 最后一条 user 消息。

        Raises:
            ValueError: 没有 user 消息时抛出。
        """

        for message in reversed(messages):
            if message.get("role") == "user":
                return message

        raise ValueError("使用 image_paths 时，messages 中必须至少包含一条 user 消息。")

    def _normalize_image_paths(
        self,
        image_paths: ImagePath | Sequence[ImagePath],
    ) -> List[ImagePath]:
        """
        把单张图片路径或多张图片路径统一成列表。

        Args:
            image_paths: 单个路径或路径列表。

        Returns:
            List[ImagePath]: 图片路径列表。
        """

        if isinstance(image_paths, (str, Path)):
            return [image_paths]

        return list(image_paths)

    def _image_to_data_url(self, image_path: ImagePath) -> str:
        """
        把本地图片转换为 base64 data URL。

        Args:
            image_path: 本地图片路径。

        Returns:
            str: data URL，例如 data:image/jpeg;base64,...。

        Raises:
            FileNotFoundError: 图片不存在时抛出。
            ValueError: 路径不是文件或 MIME 类型不是图片时抛出。
        """

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在：{path}")

        if not path.is_file():
            raise ValueError(f"图片路径不是文件：{path}")

        media_type = self._guess_image_media_type(path)
        image_bytes = path.read_bytes()

        # base64 只是传输格式。服务端识别 data:image/... 后，会按图片而不是普通文本处理。
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{media_type};base64,{image_base64}"

    def _guess_image_media_type(self, path: Path) -> str:
        """
        根据文件扩展名推断图片 MIME 类型。

        Args:
            path: 图片文件路径。

        Returns:
            str: MIME 类型，例如 image/jpeg。

        Raises:
            ValueError: 无法识别为图片时抛出。
        """

        media_type, _ = mimetypes.guess_type(path.name)
        if media_type is None or not media_type.startswith("image/"):
            raise ValueError(f"无法识别图片 MIME 类型：{path}")

        return media_type

    def _build_generation_params(self, options: LLMCallOptions | None = None) -> Dict[str, Any]:
        """
        合并默认配置与单次调用覆盖参数。

        Args:
            options: 可选单次调用覆盖参数。为 None 时完全沿用当前客户端配置。

        Returns:
            Dict[str, Any]: 可直接传给 Responses API 的生成参数。

        Raises:
            ValueError: 覆盖参数格式不合法时抛出。
        """

        effective_model = self.config.model
        effective_temperature = self.config.temperature
        effective_max_tokens = self.config.max_tokens
        effective_reasoning_effort = self.config.reasoning_effort
        disable_reasoning = False
        disable_max_tokens = False

        if options is not None:
            if options.model is not None:
                effective_model = str(options.model).strip()
                if not effective_model:
                    raise ValueError("options.model 不能为空字符串。")

            if options.temperature is not None:
                try:
                    effective_temperature = float(options.temperature)
                except (TypeError, ValueError) as error:
                    raise ValueError("options.temperature 必须是数字。") from error

            if options.max_tokens is not None:
                try:
                    effective_max_tokens = int(options.max_tokens)
                except (TypeError, ValueError) as error:
                    raise ValueError("options.max_tokens 必须是正整数。") from error
                if effective_max_tokens < 1:
                    raise ValueError("options.max_tokens 必须大于等于 1。")

            if options.reasoning_effort is not None:
                effective_reasoning_effort = str(options.reasoning_effort).strip() or None

            disable_reasoning = options.disable_reasoning
            disable_max_tokens = options.disable_max_tokens

        request_params: Dict[str, Any] = {
            "model": effective_model,
            "temperature": effective_temperature,
        }
        request_params.update(
            self._max_output_tokens_param(
                effective_max_tokens,
                disable=disable_max_tokens,
            )
        )
        request_params.update(
            self._reasoning_param(
                effective_reasoning_effort,
                disable=disable_reasoning,
            )
        )
        return request_params

    def _max_output_tokens_param(
        self,
        max_tokens: int | None,
        *,
        disable: bool = False,
    ) -> Dict[str, int]:
        """
        构造 Responses API 的 max_output_tokens 参数。

        Args:
            max_tokens: 归一化后的最大输出 token 数。
            disable: 是否显式禁用该参数。

        Returns:
            Dict[str, int]: 如果配置了 max_tokens，就返回参数字典；否则返回空字典。
        """

        # 配置字段仍叫 max_tokens，是为了让学习项目保持概念简单；
        # 真正发送给 Responses API 时需要使用 max_output_tokens。
        if disable or max_tokens is None:
            return {}
        return {"max_output_tokens": max_tokens}

    def _reasoning_param(
        self,
        reasoning_effort: str | None,
        *,
        disable: bool = False,
    ) -> Dict[str, Any]:
        """
        构造 Responses API 的 reasoning 参数。

        Args:
            reasoning_effort: 归一化后的推理强度。
            disable: 是否显式禁用该参数。

        Returns:
            Dict[str, Any]: 如果配置了 reasoning_effort，就返回推理强度参数；否则返回空字典。
        """

        # reasoning 只适用于支持推理强度的模型或兼容服务。
        # 如果服务端返回 unsupported parameter，可在 config.py 中把 reasoning_effort 改为 None，
        # 或在单次调用里通过 disable_reasoning=True 主动关闭。
        if disable or reasoning_effort is None:
            return {}
        return {"reasoning": {"effort": reasoning_effort}}

    def _validate_messages(self, messages: List[Message]) -> None:
        """
        校验 messages 的基础格式。

        Args:
            messages: 待校验的消息列表。

        Raises:
            ValueError: messages 为空、role 不合法或 content 类型不合法时抛出。
        """

        if not messages:
            raise ValueError("messages 不能为空。")

        valid_roles = {"system", "user", "assistant", "tool"}

        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")

            if role not in valid_roles:
                raise ValueError(f"第 {index} 条消息的 role 不合法：{role}。")

            if content is not None and not isinstance(content, (str, list)):
                raise ValueError(
                    f"第 {index} 条消息的 content 必须是字符串、列表或 None。"
                )

            if isinstance(content, list):
                self._validate_content_blocks(index, content)

    def _validate_content_blocks(self, message_index: int, content: List[Any]) -> None:
        """
        校验多模态 content 列表的基础格式。

        Args:
            message_index: 当前 message 在 messages 列表中的位置。
            content: message["content"] 中的多模态内容块列表。

        Raises:
            ValueError: content 为空、内容块不是 dict 或缺少 type 时抛出。
        """

        if not content:
            raise ValueError(f"第 {message_index} 条消息的 content 列表不能为空。")

        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValueError(
                    f"第 {message_index} 条消息的第 {block_index} 个内容块必须是 dict。"
                )

            block_type = block.get("type")
            if not isinstance(block_type, str) or not block_type:
                raise ValueError(
                    f"第 {message_index} 条消息的第 {block_index} 个内容块缺少 type。"
                )
