import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich import print
from typing import Optional, List
import os
import subprocess
from enum import Enum
import logging
import sys
import datetime
from time import sleep
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


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
BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR")
BASE_DATASET_DIR = os.getenv("BASE_DATASET_DIR")

# 从环境变量获取 API tokens
MODELSCOPE_TOKEN = os.getenv("MODELSCOPE_TOKEN")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 检查必要的环境变量
def check_env_vars():
    """检查必要的环境变量是否存在"""
    required_vars = {
        "BASE_MODEL_DIR": BASE_MODEL_DIR,
        "BASE_DATASET_DIR": BASE_DATASET_DIR,
        "MODELSCOPE_TOKEN": MODELSCOPE_TOKEN,
        "HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        console.print("[red]错误: 缺少必要的环境变量:[/red]")
        for var in missing_vars:
            console.print(f"[red]- {var}[/red]")
        console.print("[yellow]请确保 .env 文件存在并包含所有必要的环境变量[/yellow]")
        raise typer.Exit(1)

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
    
    console.print("\n请选择操作:")
    console.print("[1] 数据集下载")
    console.print("[2] 数据集预览")
    console.print("[0] 返回主菜单")
    
    choice = IntPrompt.ask("\n请输入选项", choices=["0", "1", "2"])
    
    if choice == 0:
        return
    elif choice == 1:
        download_dataset()
    elif choice == 2:
        preview_dataset()

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

def download_dataset():
    """数据集下载功能"""
    console.print("\n[bold cyan]数据集下载[/bold cyan]")
    
    # 选择下载源
    console.print("\n请选择下载源:")
    console.print("[1] ModelScope (默认)")
    console.print("[2] HuggingFace")
    source_choice = IntPrompt.ask("请选择", choices=["1", "2"], default="1")
    
    # 输入数据集名称
    dataset_name = Prompt.ask("\n请输入数据集名称 (例如: xiaofengalg/ShenNong_TCM_Dataset)")
    dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name.split('/')[-1])
    
    # 检查数据集是否已存在
    if check_model_exists(dataset_dir):
        if not confirm_overwrite(dataset_dir):
            console.print("[yellow]已取消下载[/yellow]")
            return
    
    # 确保目录存在
    os.makedirs(BASE_DATASET_DIR, exist_ok=True)
    
    try:
        if source_choice == 1:
            download_dataset_from_modelscope(dataset_name)
        else:
            download_dataset_from_huggingface(dataset_name)
    except Exception as e:
        console.print(f"[red]下载过程中出现错误: {str(e)}[/red]")

def download_dataset_from_modelscope(dataset_name: str, max_retries: int = 3) -> None:
    """从ModelScope下载数据集"""
    dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name.split('/')[-1])
    cmd = [
        "modelscope",
        "--token", MODELSCOPE_TOKEN,
        "download",
        "--dataset", dataset_name,
        "--local_dir", dataset_dir
    ]
    
    errors: List[str] = []
    for attempt in range(max_retries):
        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(
                    f"[yellow]从 ModelScope 下载数据集: {dataset_name} (尝试 {attempt + 1}/{max_retries})",
                    total=None
                )
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                progress.update(task, completed=100)
                
            console.print(f"\n[green]数据集已成功下载到: {dataset_dir}[/green]")
            return  # 下载成功，直接返回
        except subprocess.CalledProcessError as e:
            error_msg = f"尝试 {attempt + 1}: {str(e)}"
            if e.stderr:
                error_msg += f"\n错误输出: {e.stderr.decode('utf-8')}"
            errors.append(error_msg)
            
            if attempt < max_retries - 1:
                console.print(f"[yellow]下载失败，{5 * (attempt + 1)}秒后重试...[/yellow]")
                sleep(5 * (attempt + 1))  # 递增等待时间
            else:
                console.print("[red]达到最大重试次数，下载失败[/red]")
                for error in errors:
                    console.print(f"[red]{error}[/red]")

def download_dataset_from_huggingface(dataset_name: str, max_retries: int = 3) -> None:
    """从HuggingFace下载数据集"""
    dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name.split('/')[-1])
    cmd = [
        "huggingface-cli", "download",
        dataset_name,
        "--repo-type", "dataset",
        "--local-dir", dataset_dir,
        "--token", HUGGINGFACE_TOKEN
    ]
    
    errors: List[str] = []
    for attempt in range(max_retries):
        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(
                    f"[yellow]从 HuggingFace 下载数据集: {dataset_name} (尝试 {attempt + 1}/{max_retries})",
                    total=None
                )
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                progress.update(task, completed=100)
                
            console.print(f"\n[green]数据集已成功下载到: {dataset_dir}[/green]")
            return  # 下载成功，直接返回
        except subprocess.CalledProcessError as e:
            error_msg = f"尝试 {attempt + 1}: {str(e)}"
            if e.stderr:
                error_msg += f"\n错误输出: {e.stderr.decode('utf-8')}"
            errors.append(error_msg)
            
            if attempt < max_retries - 1:
                console.print(f"[yellow]下载失败，{5 * (attempt + 1)}秒后重试...[/yellow]")
                sleep(5 * (attempt + 1))  # 递增等待时间
            else:
                console.print("[red]达到最大重试次数，下载失败[/red]")
                for error in errors:
                    console.print(f"[red]{error}[/red]")

def preview_dataset():
    """数据集预览功能"""
    console.print("\n[bold cyan]数据集预览[/bold cyan]")
    
    # 检查数据集目录
    datasets = []
    try:
        for item in os.listdir(BASE_DATASET_DIR):
            if os.path.isdir(os.path.join(BASE_DATASET_DIR, item)):
                if not item.startswith('.'):
                    datasets.append(item)
    except FileNotFoundError:
        console.print("[red]错误: 未找到数据集目录[/red]")
        return
    
    if not datasets:
        console.print("[yellow]未找到任何可用数据集，请先下载数据集[/yellow]")
        return
    
    # 显示可用数据集列表
    console.print("\n[bold]可用数据集列表:[/bold]")
    for i, dataset in enumerate(datasets, 1):
        console.print(f"[{i}] {dataset}")
    
    # 选择数据集
    try:
        dataset_choice = int(Prompt.ask(
            "请选择数据集",
            choices=[str(i) for i in range(1, len(datasets) + 1)]
        ))
    except ValueError:
        console.print("[red]无效的选择[/red]")
        return

    selected_dataset = datasets[dataset_choice - 1]
    dataset_dir = os.path.join(BASE_DATASET_DIR, selected_dataset)
    
    # 查找并读取数据文件
    data_files = []
    file_sizes = {}
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.json', '.jsonl')):
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    data_files.append(file_path)
                    file_sizes[file_path] = file_size
                except OSError as e:
                    console.print(f"[yellow]警告: 无法获取文件大小 {file}: {str(e)}[/yellow]")
    
    if not data_files:
        console.print("[red]错误: 未找到数据文件[/red]")
        return
    
    # 按文件大小排序
    sorted_files = sorted(data_files, key=lambda x: file_sizes[x], reverse=True)
    largest_file = sorted_files[0]
    
    # 显示文件信息
    console.print("\n[bold]找到以下数据文件:[/bold]")
    for file in sorted_files:
        size_mb = file_sizes[file] / (1024 * 1024)
        is_largest = file == largest_file
        console.print(
            f"{'[green]→[/green] ' if is_largest else '  '}"
            f"{os.path.basename(file)} "
            f"({size_mb:.2f} MB)"
            f"{' [yellow](将使用此文件)[/yellow]' if is_largest else ''}"
        )
    
    # 读取数据
    import json
    data = []
    try:
        # 先尝试按行读取
        with open(largest_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 检查第一行是否是有效的 JSON
            json.loads(first_line)
            # 如果是，重新打开文件按行读取
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
    except json.JSONDecodeError:
        # 如果按行读取失败，尝试作为单个 JSON 文件读取
        with open(largest_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            if isinstance(file_data, list):
                data.extend(file_data)
            else:
                data.append(file_data)
    except Exception as e:
        console.print(f"[red]读取文件时出错: {str(e)}[/red]")
        console.print("[yellow]提示: 请确保文件是有效的 JSON 或 JSONL 格式[/yellow]")
        return
    
    if not data:
        console.print("[red]错误: 数据集为空[/red]")
        return
    
    # 显示数据集基本信息
    console.print(f"\n[bold]数据集信息:[/bold]")
    console.print(f"总记录数: {len(data)}")
    console.print(f"文件大小: {file_sizes[largest_file] / (1024 * 1024):.2f} MB")
    
    # 等待用户确认
    if not Prompt.ask("\n是否开始预览?", choices=["y", "n"], default="y").lower() == "y":
        return
    
    # 预览数据
    page_size = 10
    current_page = 0
    total_pages = (len(data) + page_size - 1) // page_size
    
    # 开始预览前清屏
    console.clear()
    console.print("\n[bold cyan]数据集预览[/bold cyan]")
    console.print("[yellow]按 Q 退出，W 上一页，E 下一页[/yellow]")
    
    while True:
        console.clear()
        console.print(f"\n[bold]数据集: {selected_dataset}[/bold]")
        console.print(f"[bold]页码: {current_page + 1}/{total_pages}[/bold]")
        
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(data))
        
        for i in range(start_idx, end_idx):
            console.print("\n" + "─" * 80)
            console.print(f"[bold cyan]记录 {i + 1}[/bold cyan]")
            console.print(json.dumps(data[i], ensure_ascii=False, indent=2))
        
        console.print("\n" + "─" * 80)
        console.print("\n[yellow]Q: 退出 | W: 上一页 | E: 下一页[/yellow]")
        
        # 根据操作系统选择合适的输入方法
        if os.name == 'nt':
            import msvcrt
            key = msvcrt.getch().decode().lower()
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if key == 'q':
            console.clear()  # 退出预览时清屏
            break
        elif key == 'w' and current_page > 0:
            current_page -= 1
        elif key == 'e' and current_page < total_pages - 1:
            current_page += 1

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """MLX CLI 主程序"""
    if ctx.invoked_subcommand is None:
        show_header()
        check_env_vars()  # 添加环境变量检查
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