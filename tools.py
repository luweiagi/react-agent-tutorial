import math
import re
import html
import urllib.request
import urllib.parse
import ssl
from datetime import datetime


class ToolExecutor:
    """工具注册中心和调度器

    每个工具由三要素定义：
    - name: 简洁唯一的标识符，供智能体在 Action 中引用
    - description: 自然语言描述，LLM 依赖它判断何时使用哪个工具
    - func: 真正执行任务的函数，接收字符串输入，返回字符串结果
    """

    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register(self, name: str, description: str, func):
        self.tools[name] = {"description": description, "func": func}

    def execute(self, tool_name: str, tool_input: str) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            available = ", ".join(self.tools.keys())
            return f"错误: 未找到工具 '{tool_name}'。可用工具: {available}"
        try:
            result = tool["func"](tool_input)
            return str(result)
        except Exception as e:
            return f"工具 '{tool_name}' 执行出错: {e}"

    def get_tools_description(self) -> str:
        lines = []
        for name, info in self.tools.items():
            lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)


def create_default_tools() -> ToolExecutor:
    """创建包含默认工具集的 ToolExecutor"""
    executor = ToolExecutor()
    executor.register(
        "Calculator",
        "计算数学表达式。输入合法的数学表达式，如 '3 * (2 + 5)' 或 'sqrt(144)'",
        calculator,
    )
    executor.register(
        "GetWeather",
        "查询指定城市的实时天气和气温。输入城市名，如 '北京' 或 'Beijing'。",
        get_weather,
    )
    executor.register(
        "Search",
        "通用网页搜索（聚合多个搜索源）。输入关键词后返回前5条结果（标题、URL和摘要），再配合 ReadWeb 阅读页面详细正文。"
        "你可以输入完整的自然语言或者核心短语进行搜索。",
        web_search,
    )
    executor.register(
        "ReadWeb",
        "读取指定网页正文。输入 URL（http/https），返回可读文本摘录",
        read_web,
    )
    executor.register(
        "GetCurrentTime",
        "获取当前日期和时间。输入任意文本即可",
        get_current_time,
    )
    return executor


# ==================== 内置工具实现 ====================


def calculator(expression: str) -> str:
    """安全地计算数学表达式，支持 math 模块中的函数"""
    # LLM 常把幂写成 ^，而 Python 的幂运算是 **。
    expression = expression.strip().replace("^", "**")
    safe_globals = {"__builtins__": {}}
    safe_globals.update({
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    })
    safe_globals["π"] = math.pi
    result = eval(expression, safe_globals)
    if isinstance(result, float) and result == int(result):
        return str(int(result))
    return str(result)

def web_search(query: str) -> str:
    """通用网页搜索，返回前 5 条结果（标题+URL+摘要）。

    输入: 任意自然语言查询，如 "爱因斯坦" 或 "2024年奥运会"
    输出: 编号列表，便于下一步让 LLM 调用 ReadWeb 逐条阅读。
    """
    query = query.strip()
    if not query:
        return "搜索词不能为空。"

    try:
        DDGS = _ddgs_class()
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"未找到与 '{query}' 相关的结果。请尝试更换关键词。"

        lines = [f"搜索结果（source: web search, query: {query}）:"]
        for i, res in enumerate(results, 1):
            title = res.get("title", "无标题")
            url = res.get("href", "无链接")
            body = res.get("body", "")
            lines.append(f"{i}. {title}\n   摘要: {body}\n   URL: {url}")

        lines.append(
            "\n请仔细阅读上述摘要，如果信息足够可以直接回答。如果需要更多细节，请选择 1-5 个高相关 URL，用 ReadWeb[url] 继续阅读。"
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Search 失败: {e}"


def read_web(url: str) -> str:
    """读取网页正文并返回精简文本，供 LLM 做事实提取。"""
    target = url.strip()
    if not target.startswith("http"):
        return "ReadWeb 只支持 http/https URL。"

    try:
        page = _http_get(_normalize_url(target), timeout=12)
        # 优先提取 <p> 段落（通常是最干净的正文），若太短则回退到全页提取
        text = _strip_html_to_text(" ".join(re.findall(r"(?is)<p[^>]*>.*?</p>", page)), max_chars=8000)
        if len(text) < 200:
            text = _strip_html_to_text(page, max_chars=8000)
            
        text = _clean_noise_text(text)
        if not text:
            return f"读取失败: {target}\n页面正文为空。"
        return f"页面: {target}\n正文摘录:\n{text}"
    except Exception as e:
        return f"读取失败: {target}\n错误: {e}"


def get_current_time(_: str) -> str:
    """返回当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def get_weather(location: str) -> str:
    """获取指定城市或地区的当前实时天气和预报。
    
    输入: 城市名称，如 "北京", "Shanghai", "New York"
    输出: 该地区的天气状况、气温等简明文本。
    """
    location = location.strip()
    if not location:
        return "城市名不能为空。"
    try:
        # 使用免费开源的 wttr.in 天气服务，format=3 表示只返回简洁的单行天气状态（城市: 状况 + 温度）
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=3"
        weather_info = _http_get(url, timeout=10)
        if weather_info and "Unknown location" not in weather_info:
            return f"当前 {location} 天气: {weather_info.strip()}"
        return f"未能获取 {location} 的天气，请检查城市名称是否正确。"
    except Exception as e:
        return f"获取天气失败: {e}"


# ==================== 公共辅助工具函数实现 ====================


def _http_get(url: str, timeout: int = 10) -> str:
    """统一的 HTTP GET，集中处理请求头和异常。"""
    # 创建一个不验证证书的 SSL 上下文，防止在特定网络环境（如代理/WSL）下 SSL 握手超时
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalize_url(url: str) -> str:
    """把可能包含中文的 URL 转成可请求的 ASCII URL。"""
    # 将完整 URL 拆成 scheme、netloc、path、query、fragment 五段，便于分别处理。
    parts = urllib.parse.urlsplit(url)
    # 对路径段做百分号编码，保留 /、% 等路径结构字符，避免中文/空格导致 URL 非法。
    path = urllib.parse.quote(parts.path, safe="/%:@")
    # 对查询字符串做编码，但保留 =、& 等参数分隔符，保证 query 语义不被破坏。
    query = urllib.parse.quote(parts.query, safe="=&%:@/?+")
    # 对片段（# 后内容）做编码，保留常见安全字符，提升兼容性。
    fragment = urllib.parse.quote(parts.fragment, safe="%:@/?+")
    # 将处理后的各段重新拼回标准 URL 并返回。
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def _strip_html_to_text(raw_html: str, max_chars: int = 5000) -> str:
    """把网页 HTML 粗提取为可读纯文本。"""
    # 删除脚本和样式等噪声标签
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    # 标签转空格，解码实体，再压缩空白
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def _clean_noise_text(text: str) -> str:
    """过滤教学场景里最常见的导航噪声词，提升正文可读性。"""
    noise = r"(table of contents|languages|目录|主菜单|导航|跳转到内容)"
    text = re.sub(noise, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _ddgs_class():
    """优先使用 `ddgs`（新包名）。旧包 `duckduckgo_search` 某版本曾强制走 Bing 后端，在部分网络下会返回空结果。"""
    try:
        from ddgs import DDGS  # type: ignore
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
        return DDGS
