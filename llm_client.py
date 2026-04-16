import json
import socket
import urllib.error
import urllib.request

# \033：ESC 字符（也常写成 \x1b），表示“后面是控制序列”
ANSI_GRAY = "\033[90m"  # \033[90m：设置前景色为亮黑/灰色（所以这段标题会变灰）
ANSI_ITALIC = "\033[3m"  # \033[3m：开启斜体（italic），让后续输出看起来像“思考态”
ANSI_RESET = "\033[0m"  # \033[0m：重置所有格式（颜色、粗体、斜体等），不然颜色/斜体可能会“泄漏”到后面普通文本。


class LLMClient:  # pylint: disable=too-few-public-methods
    """封装 Ollama 原生接口的 LLM 客户端，支持流式提取 thinking 过程"""

    def __init__(self, base_url="http://localhost:11434", model="qwen3.5:9b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.0, verbose: bool = True) -> str:
        """调用 Ollama API，若 verbose=True 则实时打印 thinking 和 content。"""
        url = f"{self.base_url}/api/chat"
        # 清洗非法 Unicode
        clean_msgs = [
            {**m, "content": m["content"].encode("utf-8", "replace").decode("utf-8")}
            for m in messages
        ]

        data = {
            "model": self.model,
            "messages": clean_msgs,
            "stream": True,
            "options": {"temperature": temperature}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        full_content = ""
        in_thinking = False

        try:
            # 发起 HTTP 请求，拿到响应流 resp（不是一次性完整文本，而是可以边到边读）
            # 不断读取模型返回的流式 JSON，每次提取出 thinking 和 content 两种增量，
            # 交给后面的代码去打印和拼接。
            with urllib.request.urlopen(req, timeout=60) as resp:
                # 按行读取响应。因为你在请求里设置了 "stream": True，
                # 模型会持续吐增量结果，所以这里会循环很多次。
                for line in resp:  # 可以理解为读一个大文件，resp只是大文件的引用
                    if not line.strip():
                        continue
                    # 把这一行 bytes 解码成字符串，再解析成 Python dict。
                    # 每一行通常像一个“增量包”（chunk）。
                    chunk = json.loads(line.decode("utf-8"))
                    msg = chunk.get("message", {})
                    # 取模型“思考过程”文本（如果模型/接口有返回）
                    thinking = msg.get("thinking", "")
                    # 取模型真正输出给用户看的正文增量
                    content = msg.get("content", "")

                    # 打印模型内部思考
                    if verbose and thinking:
                        if not in_thinking:
                            print(f"\n{ANSI_GRAY}[模型内部思考]{ANSI_ITALIC}", end="")
                            in_thinking = True
                        print(thinking, end="", flush=True)

                    # 打印模型输出给用户看的正文增量
                    if content:
                        if verbose and in_thinking:
                            print(f"{ANSI_RESET}\n", end="")
                            in_thinking = False
                        if verbose:
                            print(content, end="", flush=True)
                        # 把本次的模型输出内容拼接起来，得到完整的模型输出内容，
                        # 最终返回给调用者。
                        full_content += content

                if verbose and in_thinking:
                    print(ANSI_RESET, end="")
                if verbose:
                    print()
        except (urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            json.JSONDecodeError
        ) as e:
            if verbose:
                print(f"\nLLM 请求失败: {e}")

        return full_content
