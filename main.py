from llm_client import LLMClient
from tools import create_default_tools
from react_agent import ReActAgent
import os


DEMO_QUESTIONS = [
    "今天吉林省长春市当前天气怎么样？",
    "地球的半径大约是多少公里？它的表面积大约是多少平方公里？（提示：球体表面积 = 4πr²）",
    "爱因斯坦和牛顿，谁出生得更早？早了多少年？",
]


def run_demo(agent: ReActAgent):
    """运行预设的示例问题"""
    print("\n运行示例问题演示...\n")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'#'*60}")
        print(f"# 示例 {i}")
        print(f"{'#'*60}")
        try:
            agent.run(q)
        except KeyboardInterrupt:
            print("\n已中断当前示例。")
            break


def run_interactive(agent: ReActAgent):
    """交互式问答循环"""
    print("\nReAct Agent 已启动！输入问题开始对话，输入 quit 退出。\n")
    while True:
        try:
            question = input("你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue
        try:
            agent.run(question)
        except KeyboardInterrupt:
            print("\n已中断当前问题，可继续输入下一个问题。")


def main():
    # 支持环境变量覆盖
    # gemma4:26b
    # qwen3.5:9b
    model = os.getenv("OLLAMA_MODEL", "gemma4:26b")
    llm = LLMClient(
        base_url="http://localhost:11434",
        model=model,
    )

    tools = create_default_tools()
    agent = ReActAgent(llm_client=llm, tool_executor=tools, max_steps=10)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo(agent)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
