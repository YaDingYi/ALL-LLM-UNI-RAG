'''通过 DeepSeek Search API 联网搜索，返回可供知识库问答使用的网页摘要。'''

# 给搜索结果记录实际检索时间，方便区分检索时间与网页发表时间。
from datetime import datetime
# 用于判断搜索结果中的网址是否为正常的 HTTP(S) 地址。
from urllib.parse import urlsplit
# 统一按中国时间记录本次检索的时间。
from zoneinfo import ZoneInfo

# 与现有 DeepSeek 客户端使用同一个 HTTP 库，无需新增依赖。
import httpx


# 搜索接口与聊天接口属于不同端点，但使用同一份 DeepSeek API Key。
SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
# 搜索接口允许的问题长度为 1～128 个 Unicode 字符。
MAX_QUERY_LENGTH = 128
# 限制每次请求的等待时间；失败时不自动重试，避免重复发起请求。
SEARCH_TIMEOUT = 20


class WebSearchError(Exception):
    '''可安全展示给用户的联网搜索错误，不包含 API Key 或接口原始响应。'''


def _clean_text(value, max_length):
    '''去除多余空白并限制网页数据长度，不把网页内容当作指令。'''
    # 非字符串值按空文本处理，避免意外把结构化内容混入提示词。
    if not isinstance(value, str):
        return ""
    # 把换行、制表符等连续空白压缩成单个空格，限制显示长度。
    return " ".join(value.split())[:max_length]


def _valid_url(value):
    '''只保留可供用户查看的 HTTP(S) URL，不请求该网址。'''
    # URL 必须是合理长度的字符串，且不能包含空白或控制字符。
    if not isinstance(value, str) or not value or len(value) > 4096:
        return False
    if any(character.isspace() or ord(character) < 32 for character in value):
        return False
    try:
        # 将 URL 拆成协议、主机名和路径，避免把任意文本当作链接。
        parsed = urlsplit(value)
        # 拒绝带用户名或密码的链接，避免把凭据展示在页面或历史中。
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.hostname) and parsed.username is None and parsed.password is None
    except ValueError:
        # 格式异常的 URL 不参与生成，也不会显示在网页上。
        return False


def search_web(question, api_key, count=3):
    '''搜索当前问题，返回网页标题、URL、摘要和可选发布时间。

    成功但无结果时返回空列表；网络、认证或响应异常抛出 WebSearchError。
    本函数不会抓取结果链接的网页正文，也不会发送对话历史或本地资料。
    '''
    # 搜索只接收当前问题，检查类型和有效内容。
    if not isinstance(question, str) or not question.strip():
        raise WebSearchError("请输入要联网搜索的问题。")
    # 去除首尾空格，避免用户输入的空白占用搜索接口长度。
    query = question.strip()
    # 不静默截断问题，否则可能把用户的问题改成另一个意思。
    if len(query) > MAX_QUERY_LENGTH:
        raise WebSearchError("联网搜索的问题不能超过 128 个字符，请缩短后重试。")
    # API Key 来自现有配置；不打印、保存或放入搜索结果。
    if not isinstance(api_key, str) or not api_key.strip():
        raise WebSearchError("未配置 DeepSeek API Key，无法联网搜索。")
    # DeepSeek Search API 的 count 范围为 1～50，布尔值不能作为结果数。
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 50:
        raise WebSearchError("联网搜索结果数量必须为 1～50。")

    try:
        # 向官方搜索端点发送当前问题；need_content=False 表示只取搜索摘要。
        response = httpx.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            json={"query": query, "count": count, "need_content": False},
            timeout=SEARCH_TIMEOUT,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        # 超时不展示原始异常，以免暴露请求配置。
        raise WebSearchError("联网搜索超时，请检查网络后重试。") from None
    except httpx.RequestError:
        # DNS、连接或 TLS 问题统一使用可阅读的安全提示。
        raise WebSearchError("无法连接联网搜索服务，请检查网络后重试。") from None

    # 不输出接口的原始错误正文，避免泄露请求或账户信息。
    if response.status_code == 401:
        raise WebSearchError("联网搜索认证失败，请检查 DeepSeek API Key。")
    if response.status_code == 402:
        raise WebSearchError("联网搜索账户余额不足，请检查 DeepSeek 账户。")
    if response.status_code == 429:
        raise WebSearchError("联网搜索请求过于频繁，请稍后重试。")
    if response.status_code != 200:
        raise WebSearchError(f"联网搜索服务暂不可用（HTTP {response.status_code}），请稍后重试。")

    try:
        # 官方接口成功时返回包含 results 列表的 JSON 对象。
        payload = response.json()
    except ValueError:
        # 非 JSON 内容不是可使用的搜索结果。
        raise WebSearchError("联网搜索服务返回了无法解析的数据。") from None
    # 不把格式错误误判成“搜索成功但没有结果”。
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise WebSearchError("联网搜索服务返回的数据格式不正确。")

    # 对一次搜索使用相同的获取时间，不把它误写成网页发布时间。
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    # 保存清洗后的搜索结果，并按 URL 去重。
    results = []
    seen_urls = set()
    # 按搜索服务的原有顺序处理，不依据数值自行重新排序。
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        # 不读取网页原文；只使用服务返回的标题、URL 和查询摘要。
        url = item.get("url")
        if not _valid_url(url) or url in seen_urls:
            continue
        # summary 更适合模型阅读；不存在时使用较短的 snippet。
        summary = _clean_text(item.get("summary"), 1200) or _clean_text(item.get("snippet"), 1200)
        if not summary:
            continue
        # 缺少标题时给出中性名称，避免展示空白链接。
        title = _clean_text(item.get("title"), 200) or "未提供标题"
        # 仅在服务确实提供时间时保存，不能自行推测发布日期。
        date = _clean_text(item.get("date"), 80) or None
        # 标记本条记录来自互联网，与本地文件资料区分。
        results.append({"type": "web", "source": title, "url": url, "text": summary, "page": None, "date": date, "retrieved_at": retrieved_at})
        seen_urls.add(url)
        # 即使接口返回过多结果，也只交给上层最多 count 条。
        if len(results) >= count:
            break
    # 接口明明返回了记录却没有任何可用摘要，应提示响应异常。
    if payload["results"] and not results:
        raise WebSearchError("联网搜索返回的结果缺少有效链接或摘要。")
    # 空结果是正常搜索结果，由调用方决定如何向用户提示。
    return results
