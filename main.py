import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich import print
from typing import Optional
import os
import subprocess
from enum import Enum
import logging
import sys
import datetime

class ToggleableRichHandler(RichHandler):
    """可切换显示的 RichHandler"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = False

    def emit(self, record):
        if self.enabled:
            super().emit(record)

# 配置日志
show_logs = False  # 控制日志显示
log_file = f"mlx-cli-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
rich_handler = ToggleableRichHandler(
    rich_tracebacks=True,
    show_time=False,
    show_path=False
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
    handlers=[
        rich_handler,
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger("mlx-cli")

class DownloadSource(str, Enum):
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"

# 初始化
app = typer.Typer(
    name="MLX CLI",
    help="MLX框架模型训练和推理工具",
    add_completion=False
)

console = Console()
BASE_MODEL_DIR = "/Users/wyek1n/Downloads/MLX/model"

def show_header():
    """显示欢迎信息"""
    console.print(Panel.fit(
        "[bold blue]欢迎使用 MLX CLI 工具[/bold blue]\n"
        "版本: 0.1.0",
        title="MLX CLI",
        border_style="blue"
    ))

def check_environment():
    """检查运行环境"""
    try:
        import mlx
        console.print("[green]✓[/green] MLX 环境检查通过")
        try:
            from mlx_lm import load
            console.print("[green]✓[/green] MLX-LM 环境检查通过")
        except ImportError:
            console.print("[red]✗[/red] 请先安装 MLX-LM 包")
            console.print("[yellow]运行: pip install mlx-lm[/yellow]")
            raise typer.Exit(1)
    except ImportError:
        console.print("[red]✗[/red] 请先安装 MLX 框架")
        raise typer.Exit(1)

def show_menu():
    """显示主菜单"""
    console.print("\n[bold cyan]请选择操作:[/bold cyan]")
    console.print("[1] 模型下载")
    console.print("[2] 数据准备")
    console.print("[3] 模型微调")
    console.print("[4] 模型合并")
    console.print("[5] 模型评估")
    console.print("[6] 模型对话")
    console.print(f"[7] {'隐藏' if show_logs else '显示'}日志")
    console.print("[0] 退出程序")
    
    choice = IntPrompt.ask("\n请输入选项", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
    return choice

def check_model_exists(model_dir: str) -> bool:
    """检查模型是否已存在"""
    return os.path.exists(model_dir)

def confirm_overwrite(model_dir: str) -> bool:
    """确认是否覆盖已存在的模型"""
    console.print(f"\n[yellow]警告: 模型目录已存在: {model_dir}[/yellow]")
    return Prompt.ask("是否覆盖?", choices=["y", "n"], default="n").lower() == "y"

def download_model():
    """模型下载功能"""
    console.print("\n[bold cyan]模型下载[/bold cyan]")
    
    # 选择下载源
    console.print("\n请选择下载源:")
    console.print("[1] ModelScope (默认)")
    console.print("[2] HuggingFace")
    source_choice = IntPrompt.ask("请选择", choices=["1", "2"], default="1")
    
    # 输入模型名称
    model_name = Prompt.ask("\n请输入模型名称 (例如: Qwen/Qwen2.5-0.5B-Instruct)")
    model_dir = os.path.join(BASE_MODEL_DIR, model_name.split('/')[-1])
    
    # 检查模型是否已存在
    if check_model_exists(model_dir):
        if not confirm_overwrite(model_dir):
            console.print("[yellow]已取消下载[/yellow]")
            return
    
    # 确保目录存在
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    
    try:
        if source_choice == 1:
            download_from_modelscope(model_name)
        else:
            download_from_huggingface(model_name)
    except Exception as e:
        console.print(f"[red]下载过程中出现错误: {str(e)}[/red]")

def prepare_data():
    """数据准备功能"""
    console.print("\n[bold cyan]数据准备[/bold cyan]")
    # TODO: 实现数据准备功能
    console.print("[yellow]该功能正在开发中...[/yellow]")

def fine_tune():
    """模型微调功能"""
    console.print("\n[bold cyan]模型微调[/bold cyan]")
    # TODO: 实现模型微调功能
    console.print("[yellow]该功能正在开发中...[/yellow]")

def merge_model():
    """模型合并功能"""
    console.print("\n[bold cyan]模型合并[/bold cyan]")
    # TODO: 实现模型合并功能
    console.print("[yellow]该功能正在开发中...[/yellow]")

def evaluate_model():
    """模型评估功能"""
    console.print("\n[bold cyan]模型评估[/bold cyan]")
    # TODO: 实现模型评估功能
    console.print("[yellow]该功能正在开发中...[/yellow]")

def chat_with_model():
    """模型对话功能"""
    console.print("\n[bold cyan]模型对话[/bold cyan]")
    
    try:
        log.info("=== 开始模型对话功能 ===")
        # 检查模型目录
        models = []
        log.info("开始扫描模型目录")
        try:
            for item in os.listdir(BASE_MODEL_DIR):
                if os.path.isdir(os.path.join(BASE_MODEL_DIR, item)):
                    # 过滤掉特殊目录
                    if not item.startswith('.') and item != 'venv':
                        log.debug(f"找到模型目录: {item}")
                        models.append(item)
        except FileNotFoundError as e:
            log.error(f"模型目录不存在: {BASE_MODEL_DIR}", exc_info=True)
            console.print("[red]错误: 未找到模型目录[/red]")
            return
        
        if not models:
            log.warning("未找到任何可用模型")
            console.print("[yellow]未找到任何可用模型，请先下载模型[/yellow]")
            return
        
        log.info(f"共找到 {len(models)} 个可用模型")
        # 显示可用模型列表
        console.print("\n[bold]可用模型列表:[/bold]")
        for i, model in enumerate(models, 1):
            console.print(f"[{i}] {model}")
        
        # 选择模型
        try:
            model_choice = int(Prompt.ask(
                "请选择模型",
                choices=[str(i) for i in range(1, len(models) + 1)]
            ))
            log.info(f"用户选择了模型 {models[model_choice - 1]}")
        except ValueError:
            log.error("用户输入了无效的选择")
            console.print("[red]无效的选择[/red]")
            return

        selected_model = models[model_choice - 1]
        model_path = os.path.join(BASE_MODEL_DIR, selected_model)
        
        log.info(f"正在加载模型: {model_path}")
        console.print(f"\n[yellow]正在加载模型 {selected_model}...[/yellow]")
        console.print(f"模型路径: {model_path}")
        
        try:
            # 确保mlx_lm已安装
            try:
                log.debug("正在导入 mlx_lm")
                from mlx_lm import load, stream_generate
            except ImportError:
                log.error("mlx_lm 包未安装")
                console.print("[red]错误: 请先安装 mlx-lm 包[/red]")
                console.print("[yellow]运行: pip install mlx-lm[/yellow]")
                return

            # 尝试加载模型
            log.info("开始初始化模型")
            console.print("[yellow]正在初始化模型...[/yellow]")
            log.debug("调用 load() 函数")
            try:
                model, tokenizer = load(model_path)
                log.info("模型加载成功")
            except Exception as e:
                log.error(f"模型加载失败: {str(e)}", exc_info=True)
                raise
            console.print("[green]模型加载成功！[/green]")
            
            log.debug("初始化 Chatbot 类")
            class Chatbot:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.messages = [{"role": "system", "content": "我是一个AI助手"}]
                    log.debug("Chatbot 初始化完成")

                def add_user_message(self, content):
                    self.messages.append({"role": "user", "content": content})

                def get_response(self):
                    chat_template = self.tokenizer.apply_chat_template(
                        self.messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    return stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt=chat_template,
                        max_tokens=512
                    )

                def chat(self, user_input):
                    log.debug(f"收到用户输入: {user_input}")
                    answer = ""
                    self.add_user_message(user_input)
                    response = self.get_response()
                    for text in response:
                        text_content = text.text if hasattr(text, 'text') else str(text)
                        answer = answer + text_content
                        yield text_content
                    self.messages.append({"role": "assistant", "content": answer})
                    log.debug("完成一轮对话")
            
            # 创建聊天机器人实例
            log.info("创建 Chatbot 实例")
            chatbot = Chatbot(model, tokenizer)
            console.print("[green]开始对话 (输入 'exit' 或 '退出' 结束对话)[/green]\n")
            
            while True:
                try:
                    user_input = Prompt.ask("\n[bold cyan]用户[/bold cyan]")
                    if user_input.lower() in ["退出", "exit"]:
                        log.info("用户退出对话")
                        break
                        
                    console.print("[bold green]助手[/bold green]", end=": ")
                    response = chatbot.chat(user_input)
                    for text in response:
                        if text != "":
                            print(text, end="", flush=True)
                    console.print()
                except Exception as e:
                    log.exception("对话过程中出现错误")
                    console.print(f"\n[red]对话出错: {str(e)}[/red]")
                    continue
            
        except Exception as e:
            log.exception("模型初始化过程中出现错误")
            console.print(f"[red]初始化过程中出现错误: {str(e)}[/red]")
            console.print("[yellow]请确保模型文件完整且格式正确[/yellow]")
            console.print(f"[yellow]详细错误日志已保存到: {log_file}[/yellow]")
            return
    finally:
        log.info("=== 结束模型对话功能 ===")

def download_from_modelscope(model_name: str) -> None:
    """从ModelScope下载模型"""
    model_dir = os.path.join(BASE_MODEL_DIR, model_name.split('/')[-1])
    cmd = [
        "modelscope", "download",
        "--model", model_name,
        "--local_dir", model_dir
    ]
    
    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True  # 这将使进度条在完成后消失
        ) as progress:
            task = progress.add_task(f"[yellow]从 ModelScope 下载模型: {model_name}", total=None)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            progress.update(task, completed=100)
            
        console.print(f"\n[green]模型已成功下载到: {model_dir}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]下载失败: {str(e)}[/red]")

def download_from_huggingface(model_name: str) -> None:
    """从HuggingFace下载模型"""
    model_dir = os.path.join(BASE_MODEL_DIR, model_name.split('/')[-1])
    cmd = [
        "huggingface-cli", "download",
        model_name,
        "--local-dir", model_dir
    ]
    
    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True  # 这将使进度条在完成后消失
        ) as progress:
            task = progress.add_task(f"[yellow]从 HuggingFace 下载模型: {model_name}", total=None)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            progress.update(task, completed=100)
            
        console.print(f"\n[green]模型已成功下载到: {model_dir}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]下载失败: {str(e)}[/red]")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """MLX CLI 主程序"""
    if ctx.invoked_subcommand is None:
        show_header()
        check_environment()
        
        global show_logs, rich_handler
        while True:
            choice = show_menu()
            
            if choice == 0:
                console.print("[cyan]感谢使用,再见![/cyan]")
                break
            elif choice == 1:
                download_model()
            elif choice == 2:
                prepare_data()
            elif choice == 3:
                fine_tune()
            elif choice == 4:
                merge_model()
            elif choice == 5:
                evaluate_model()
            elif choice == 6:
                chat_with_model()
            elif choice == 7:
                show_logs = not show_logs
                rich_handler.enabled = show_logs
                console.print(f"[green]已{'隐藏' if not show_logs else '显示'}日志[/green]")

if __name__ == "__main__":
    app() 