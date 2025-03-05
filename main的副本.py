import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich import print
from typing import Optional, List, Any, Dict
import os
import subprocess
from enum import Enum
import logging
import sys
from time import sleep
from dotenv import load_dotenv
import json
import yaml
from pathlib import Path
import wandb
from datetime import datetime
from threading import Thread
from queue import Queue
import signal
from contextlib import contextmanager

# 加载环境变量
load_dotenv()

# 创建logs目录
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

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
log_file = os.path.join(LOGS_DIR, f"mlx-cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
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
    console.print("[3] 数据集转换")
    console.print("[4] 数据集分割")
    console.print("[0] 返回主菜单")
    
    choice = IntPrompt.ask("\n请输入选项", choices=["0", "1", "2", "3", "4"])
    
    if choice == 0:
        return
    elif choice == 1:
        download_dataset()
    elif choice == 2:
        preview_dataset()
    elif choice == 3:
        convert_dataset()
    elif choice == 4:
        split_dataset()

def fine_tune():
    """模型微调功能"""
    console.print("\n[bold cyan]模型微调[/bold cyan]")
    
    # 检查 wandb API key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        console.print("[yellow]警告: 未找到WANDB_API_KEY，将不会记录训练过程[/yellow]")
        return
    
    try:
        # 收集所有训练参数
        params = collect_training_params()
        if not params:
            return
            
        # 初始化 wandb
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{params['model'].split('/')[-1]}_{timestamp}"
        
        wandb.init(
            project="mlx-finetune",
            name=run_name,
            config=params
        )
        
        # 定义要追踪的指标
        wandb.define_metric("train/global_step", summary="max")
        wandb.define_metric("train/epoch", summary="max")
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("train/grad_norm", summary="mean")
        wandb.define_metric("train/learning_rate", summary="last")
        
        console.print(f"[green]wandb run 初始化成功: {run_name}[/green]")
        
        # 构建训练命令
        cmd = build_training_command(params)
        
        # 显示命令预览
        console.print("\n[bold]将执行以下命令:[/bold]")
        console.print(" ".join(cmd))
        
        # 确认执行
        if not Prompt.ask("\n是否开始训练?", choices=["y", "n"], default="y").lower() == "y":
            console.print("[yellow]已取消训练[/yellow]")
            return
        
        # 执行训练
        run_training(cmd)
        
    except Exception as e:
        console.print(f"[red]训练过程出错: {str(e)}[/red]")
    finally:
        if wandb.run is not None:
            wandb.finish()

def collect_training_params():
    """收集训练参数"""
    params = {}
    try:
        # 选择模型
        models = []
        try:
            for item in os.listdir(BASE_MODEL_DIR):
                if os.path.isdir(os.path.join(BASE_MODEL_DIR, item)):
                    models.append(item)
        except FileNotFoundError:
            console.print("[red]错误: 未找到模型目录[/red]")
            return None
        
        if not models:
            console.print("[yellow]未找到任何可用模型，请先下载模型[/yellow]")
            return None
        
        console.print("\n[bold]可用模型列表:[/bold]")
        for i, model in enumerate(models, 1):
            console.print(f"[{i}] {model}")
        
        try:
            model_choice = int(Prompt.ask(
                "请选择模型",
                choices=[str(i) for i in range(1, len(models) + 1)]
            ))
            selected_model = models[model_choice - 1]
            params["model"] = os.path.join(BASE_MODEL_DIR, selected_model)
            
            # 根据选择的模型设置adapter保存路径
            adapter_base_dir = "/Users/wyek1n/Downloads/MLX/adapter"
            adapter_model_dir = os.path.join(adapter_base_dir, selected_model)
            os.makedirs(adapter_model_dir, exist_ok=True)
            params["adapter_path"] = adapter_model_dir
            
            # 如果需要恢复训练，使用模型特定的权重文件路径
            if Prompt.ask(
                "\n是否需要恢复之前的训练?",
                choices=["y", "n"],
                default="n"
            ).lower() == "y":
                params["resume_adapter_file"] = os.path.join(adapter_model_dir, "adapter_weights.safetensors")
            
            # 选择微调类型
            console.print("\n[bold]请选择微调类型:[/bold]")
            console.print("[1] LoRA (默认)")
            console.print("[2] DoRA")
            console.print("[3] Full")
            
            fine_tune_types = {1: "lora", 2: "dora", 3: "full"}
            type_choice = int(Prompt.ask("请选择", choices=["1", "2", "3"], default="1"))
            params["fine_tune_type"] = fine_tune_types[type_choice]
            
            # 设置其他参数
            params["num_layers"] = int(Prompt.ask(
                "请输入微调层数 (-1表示全部层)",
                default="16"
            ))
            
            params["batch_size"] = int(Prompt.ask(
                "请输入批次大小",
                default="1"
            ))
            
            params["iters"] = int(Prompt.ask(
                "请输入训练迭代次数",
                default="1000"
            ))
            
            params["learning_rate"] = float(Prompt.ask(
                "请输入学习率",
                default="0.001"
            ))
            
            params["steps_per_report"] = int(Prompt.ask(
                "请输入损失报告间隔步数",
                default="10"
            ))
            
            return params
        except ValueError:
            console.print("[red]无效的选择[/red]")
            return None
    except Exception as e:
        console.print(f"[red]参数收集出错: {str(e)}[/red]")
        return None

def build_training_command(params):
    """构建训练命令"""
    cmd = ["mlx_lm.lora", "--train"]
    for k, v in params.items():
        if v is not None:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    cmd.extend(["--data", "/Users/wyek1n/Downloads/Code/MLX/MLX-CLI/lora/data"])
    return cmd

@contextmanager
def handle_training_interrupt():
    """处理训练过程中的中断信号"""
    original_handlers = {}
    training_process = None
    
    def handle_signal(signum, frame):
        if training_process:
            console.print("\n[yellow]正在终止训练进程...[/yellow]")
            training_process.terminate()
            training_process.wait()
        # 恢复原始信号处理器
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
        raise KeyboardInterrupt
    
    try:
        # 保存原始信号处理器
        original_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTSTP: signal.getsignal(signal.SIGTSTP)
        }
        # 设置新的信号处理器
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTSTP, handle_signal)
        yield lambda p: setattr(sys.modules[__name__], 'training_process', p)
    finally:
        # 恢复原始信号处理器
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)

def run_training(cmd):
    """执行训练过程"""
    try:
        with handle_training_interrupt() as set_process:
            env = os.environ.copy()
            env["WANDB_CONSOLE"] = "off"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env,
                bufsize=1,
                text=True
            )
            set_process(process)
            
            metrics_queue = Queue()
            
            def process_metrics(queue):
                while True:
                    metrics = queue.get()
                    if metrics is None:
                        break
                    try:
                        wandb.log(metrics)
                    except Exception as e:
                        log.warning(f"wandb记录出错: {str(e)}")
            
            # 启动指标处理线程
            metrics_thread = Thread(target=process_metrics, daemon=True, args=(metrics_queue,))
            metrics_thread.start()

            # 实时读取并处理输出
            while process.poll() is None:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                for line in (stdout_line, stderr_line):
                    if line:
                        line = line.strip()
                        # 直接打印到控制台
                        print(line, flush=True)
                        # 解析并记录指标
                        metrics = parse_training_output(line)
                        if metrics:
                            metrics_queue.put(metrics)
            
            # 处理剩余输出
            stdout, stderr = process.communicate()
            for line in (stdout.splitlines() + stderr.splitlines()):
                if line.strip():
                    print(line.strip(), flush=True)
                    metrics = parse_training_output(line.strip())
                    if metrics:
                        metrics_queue.put(metrics)
            
            # 停止指标处理线程
            metrics_queue.put(None)
            metrics_thread.join()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]训练已被用户中断[/yellow]")
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        raise Exception(f"训练进程出错: {str(e)}")

def parse_training_output(output: str) -> dict:
    """解析训练输出并提取关键信息"""
    try:
        console.print(f"[yellow]正在解析训练输出: {output}[/yellow]")  # 打印原始输出
        # MLX-LM 的输出格式示例：
        # Training: iter = 10/1000: loss = 2.5, grad_norm = 0.1, lr = 0.001
        metrics = {}
        
        if "Training:" in output and "loss" in output:
            # 去除 "Training:" 前缀
            output = output.split("Training:")[-1].strip()
            
            # 提取步数/迭代次数
            if "iter =" in output:
                iter_part = output.split("iter =")[1].split(":")[0].strip()
            else:
                iter_part = output.split("step =")[1].split(":")[0].strip()
            
            current_iter = int(iter_part.split("/")[0])
            total_iters = int(iter_part.split("/")[1])
            
            # 提取训练指标
            metrics_part = output.split(":")[-1].strip()
            for metric in metrics_part.split(","):
                if "=" in metric:
                    key, value = [x.strip() for x in metric.split("=")]
                    try:
                        value = float(value)
                        if key == "lr":
                            key = "learning_rate"
                        metrics[f"train/{key}"] = value
                    except ValueError:
                        continue
            
            # 添加步数和 epoch 信息
            metrics["train/global_step"] = current_iter
            metrics["train/epoch"] = current_iter / total_iters
            
            console.print(f"[green]成功解析指标: {metrics}[/green]")  # 打印解析结果
            return metrics
    except Exception as e:
        console.print(f"[red]解析训练输出时出错: {str(e)}[/red]")
    console.print("[yellow]未能解析指标[/yellow]")
    return {}

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

def detect_file_format(file_path: str) -> str:
    """检测文件格式
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 'json' 或 'jsonl'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            
            # 如果第二行存在且是有效的JSON，说明是JSONL格式
            if second_line and json.loads(first_line) and json.loads(second_line):
                return 'jsonl'
            
            # 重新打开文件尝试作为单个JSON读取
            f.seek(0)
            json.load(f)
            return 'json'
    except json.JSONDecodeError:
        # 如果作为单个JSON读取失败，再次尝试按行读取
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        json.loads(line.strip())
                return 'jsonl'
        except json.JSONDecodeError:
            raise ValueError("无法识别文件格式")
    except Exception as e:
        raise ValueError(f"读取文件时出错: {str(e)}")

def convert_to_jsonl(file_path: str) -> str:
    """将JSON文件转换为JSONL格式
    
    Args:
        file_path: 源文件路径
        
    Returns:
        str: 转换后的JSONL文件路径
    """
    jsonl_path = file_path.rsplit('.', 1)[0] + '.jsonl'
    
    try:
        # 首先检测文件格式
        file_format = detect_file_format(file_path)
        
        # 如果已经是JSONL格式，只需要重命名
        if file_format == 'jsonl':
            if not file_path.endswith('.jsonl'):
                os.rename(file_path, jsonl_path)
                console.print(f"[green]已将 {os.path.basename(file_path)} 重命名为JSONL格式[/green]")
            return jsonl_path
        
        # 如果是JSON格式，进行转换
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 确保数据是列表
        if not isinstance(data, list):
            data = [data]
            
        # 写入JSONL文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 删除原文件
        os.remove(file_path)
        console.print(f"[green]已将 {os.path.basename(file_path)} 转换为JSONL格式[/green]")
        return jsonl_path
    except Exception as e:
        console.print(f"[red]转换文件 {file_path} 时出错: {str(e)}[/red]")
        if os.path.exists(jsonl_path) and jsonl_path != file_path:
            os.remove(jsonl_path)  # 清理未完成的转换文件
        return file_path  # 转换失败时返回原文件路径

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
    
    # 如果最大文件不是JSONL格式，进行转换
    if not largest_file.endswith('.jsonl'):
        console.print("\n[yellow]检测到非JSONL格式文件，正在转换...[/yellow]")
        largest_file = convert_to_jsonl(largest_file)
        # 更新文件大小信息
        file_sizes[largest_file] = os.path.getsize(largest_file)
    
    # 显示文件信息
    console.print("\n[bold]找到以下数据文件:[/bold]")
    for file in sorted_files:
        if file == largest_file or file.endswith('.jsonl'):  # 只显示JSONL文件
            size_mb = file_sizes[file] / (1024 * 1024)
            is_largest = file == largest_file
            console.print(
                f"{'[green]→[/green] ' if is_largest else '  '}"
                f"{os.path.basename(file)} "
                f"({size_mb:.2f} MB)"
                f"{' [yellow](将使用此文件)[/yellow]' if is_largest else ''}"
            )
    
    # 读取数据
    data = []
    try:
        with open(largest_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
    except Exception as e:
        console.print(f"[red]读取文件时出错: {str(e)}[/red]")
        console.print("[yellow]提示: 请确保文件是有效的 JSONL 格式[/yellow]")
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

class MLXDataConverter:
    """MLX数据集转换器"""
    # 定义字段映射
    INPUT_FIELDS = ["question", "query", "instruction", "prompt", "task"]  # 主输入
    CONTEXT_FIELDS = ["context", "input", "metadata", "example", "evidence", "schema"]  # 上下文
    OUTPUT_FIELDS = ["answer", "response", "output", "result", "solution", "completion"]  # 输出

    def __init__(self, input_data: Any, target_format: str = None, system_message: str = None):
        self.input_data = input_data
        self.target_format = target_format
        self.system_message = system_message
        self.format_prefixes = {
            "chat": {
                "context": "",
                "input": ""
            },
            "completions": {
                "context": "",
                "input": ""
            },
            "text": {
                "context": "",
                "input": ""
            }
        }

    def detect_format(self) -> str:
        """自动检测输入数据特征并推断目标格式"""
        if isinstance(self.input_data, list) and all("role" in item for item in self.input_data):
            return "chat"
        elif isinstance(self.input_data, dict):
            if any(f in self.input_data for f in self.INPUT_FIELDS):
                return "completions"
        return "text"

    def _get_prefixes(self) -> Dict[str, str]:
        """获取当前格式的前缀配置"""
        format_to_use = self.target_format or self.detect_format()
        return self.format_prefixes.get(format_to_use, {"context": "", "input": ""})

    def convert_to_chat(self) -> Dict[str, list]:
        """转换为chat格式"""
        messages = []
        
        if isinstance(self.input_data, dict):
            input_text = next((self.input_data[f] for f in self.INPUT_FIELDS if f in self.input_data), "")
            context_text = next((self.input_data[f] for f in self.CONTEXT_FIELDS if f in self.input_data), "")
            completion = next((self.input_data[f] for f in self.OUTPUT_FIELDS if f in self.input_data), "")
            
            full_prompt = ""
            prefixes = self._get_prefixes()
            if context_text:
                full_prompt += f"{prefixes['context']}{context_text}\n\n"
            if input_text:
                full_prompt += f"{prefixes['input']}{input_text}"
            
            if full_prompt:
                messages = [
                    {"role": "user", "content": full_prompt.strip()},
                    {"role": "assistant", "content": completion}
                ]
        else:
            messages = [{"role": "user", "content": str(self.input_data)}]
            
        # 添加system message
        if self.system_message is not None:
            messages.insert(0, {
                "role": "system",
                "content": self.system_message
            })
            
        return {"messages": messages}

    def convert_to_completions(self) -> Dict[str, str]:
        """转换为completions格式"""
        if isinstance(self.input_data, dict):
            input_text = next((self.input_data[f] for f in self.INPUT_FIELDS if f in self.input_data), "")
            context_text = next((self.input_data[f] for f in self.CONTEXT_FIELDS if f in self.input_data), "")
            output_text = next((self.input_data[f] for f in self.OUTPUT_FIELDS if f in self.input_data), "")
            
            full_prompt = ""
            if context_text:
                full_prompt += f"{context_text}\n\n"
            if input_text:
                full_prompt += input_text
            
            # 添加system message到prompt开头
            if self.system_message:
                full_prompt = f"{self.system_message}\n{full_prompt}"
            
            return {"prompt": full_prompt.strip(), "completion": output_text}
        return {"prompt": str(self.input_data), "completion": ""}

    def convert_to_text(self) -> Dict[str, str]:
        """转换为text格式"""
        if isinstance(self.input_data, dict):
            # 提取字段
            input_text = next((self.input_data[f] for f in self.INPUT_FIELDS if f in self.input_data), "")
            context_text = next((self.input_data[f] for f in self.CONTEXT_FIELDS if f in self.input_data), "")
            output_text = next((self.input_data[f] for f in self.OUTPUT_FIELDS if f in self.input_data), "")
            
            # 构造完整文本
            text = ""
            if context_text:
                text += f"{context_text}\n\n"
            if input_text:
                text += f"{input_text}\n\n"
            text += output_text
        else:
            text = str(self.input_data)
        return {"text": text}

    def convert(self) -> Dict[str, Any]:
        """主转换函数"""
        format_to_use = self.target_format or self.detect_format()
        if format_to_use == "chat":
            return self.convert_to_chat()
        elif format_to_use == "completions":
            return self.convert_to_completions()
        else:
            return self.convert_to_text()

def convert_dataset():
    """数据集转换功能"""
    console.print("\n[bold cyan]数据集转换[/bold cyan]")
    
    # 检查数据集目录
    datasets = []
    try:
        for item in os.listdir(BASE_DATASET_DIR):
            if (os.path.isdir(os.path.join(BASE_DATASET_DIR, item)) 
                and not item.startswith('MLX_') 
                and not item.startswith('.')):  # 过滤掉隐藏目录
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
    source_dir = os.path.join(BASE_DATASET_DIR, selected_dataset)
    
    # 选择目标格式
    console.print("\n[bold]请选择转换格式:[/bold]")
    console.print("[1] Completions 格式")
    console.print("[2] Chat 格式")
    console.print("[3] Text 格式")
    
    try:
        format_choice = int(Prompt.ask(
            "请选择格式",
            choices=["1", "2", "3"]
        ))
    except ValueError:
        console.print("[red]无效的选择[/red]")
        return
    
    # 映射选择到格式
    format_map = {
        1: "completions",
        2: "chat",
        3: "text"
    }
    target_format = format_map[format_choice]
    
    # 如果选择了completions或chat格式，请求system message
    system_message = None
    if target_format in ["completions", "chat"]:
        console.print("\n[bold]请输入系统提示词：[/bold]")
        system_message = Prompt.ask("系统提示词", show_default=False)
        if not system_message.strip():
            system_message = None
    
    # 根据选择的格式设置目标目录名称
    format_prefix = target_format.capitalize()
    target_dir = os.path.join(BASE_DATASET_DIR, f"MLX_{format_prefix}_{selected_dataset}")
    
    # 查找所有 JSON/JSONL 文件
    data_files = []
    file_sizes = {}
    for root, _, files in os.walk(source_dir):
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
    
    # 转换所有非JSONL文件
    jsonl_files = []
    for file_path in sorted_files:
        if not file_path.endswith('.jsonl'):
            console.print(f"\n[yellow]正在将 {os.path.basename(file_path)} 转换为JSONL格式...[/yellow]")
            try:
                jsonl_path = convert_to_jsonl(file_path)
                jsonl_files.append(jsonl_path)
            except Exception as e:
                console.print(f"[red]转换失败: {str(e)}[/red]")
                continue
        else:
            jsonl_files.append(file_path)
    
    if not jsonl_files:
        console.print("[red]错误: 没有可用的JSONL文件[/red]")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 转换数据集
    for source_file in jsonl_files:
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        console.print(f"\n[yellow]正在转换: {os.path.basename(source_file)} -> {format_prefix} 格式[/yellow]")
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f_in, \
                 open(target_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    data = json.loads(line.strip())
                    converter = MLXDataConverter(data, target_format, system_message)
                    converted_data = converter.convert()
                    f_out.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
            
            console.print(f"[green]✓ 已转换并保存到: {target_file}[/green]")
        except Exception as e:
            console.print(f"[red]转换失败: {str(e)}[/red]")
            continue
    
    console.print(f"\n[green]数据集已转换为 {format_prefix} 格式![/green]")
    console.print(f"[green]转换后的数据集保存在: {target_dir}[/green]")

def split_dataset():
    """数据集分割功能"""
    console.print("\n[bold cyan]数据集分割[/bold cyan]")
    
    # 检查数据集目录
    datasets = []
    try:
        for item in os.listdir(BASE_DATASET_DIR):
            if (os.path.isdir(os.path.join(BASE_DATASET_DIR, item)) 
                and item.startswith('MLX_')):  # 只显示MLX_开头的数据集
                datasets.append(item)
    except FileNotFoundError:
        console.print("[red]错误: 未找到数据集目录[/red]")
        return
    
    if not datasets:
        console.print("[yellow]未找到任何可用的转换后数据集，请先进行数据集转换[/yellow]")
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
    
    # 查找JSONL文件
    jsonl_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    if not jsonl_files:
        console.print("[red]错误: 未找到JSONL文件[/red]")
        return
    
    # 读取数据
    data = []
    for file in jsonl_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
        except Exception as e:
            console.print(f"[red]读取文件 {file} 时出错: {str(e)}[/red]")
            return
    
    total_samples = len(data)
    if total_samples == 0:
        console.print("[red]错误: 数据集为空[/red]")
        return
    
    # 询问随机抽取数量
    console.print(f"\n[bold]数据集总样本数: {total_samples}[/bold]")
    try:
        sample_size = int(Prompt.ask(
            "请输入需要随机抽取的样本数（直接回车使用全部数据）",
            default=str(total_samples)
        ))
        if not 0 < sample_size <= total_samples:
            raise ValueError("样本数超出范围")
    except ValueError as e:
        console.print(f"[red]无效的样本数: {str(e)}[/red]")
        return
    
    # 询问训练集比例
    try:
        train_ratio = float(Prompt.ask(
            "请输入训练集比例（0-1之间，默认0.9）",
            default="0.9"
        ))
        if not 0 < train_ratio <= 1:
            raise ValueError("比例必须在0-1之间")
    except ValueError as e:
        console.print(f"[red]无效的比例: {str(e)}[/red]")
        return
    
    # 询问测试集比例
    if train_ratio < 1:
        try:
            test_ratio = float(Prompt.ask(
                "请输入测试集比例（0-1之间，默认0.5）",
                default="0.5"
            ))
            if not 0 < test_ratio <= 1:
                raise ValueError("比例必须在0-1之间")
        except ValueError as e:
            console.print(f"[red]无效的比例: {str(e)}[/red]")
            return
    
    # 创建输出目录
    output_dir = "/Users/wyek1n/Downloads/Code/MLX/MLX-CLI/lora/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机抽样和分割
    import random
    random.shuffle(data)
    selected_data = data[:sample_size]
    
    # 计算各集合的大小
    train_size = int(sample_size * train_ratio)
    if train_ratio < 1:
        remaining_size = sample_size - train_size
        test_size = int(remaining_size * test_ratio)
        valid_size = remaining_size - test_size
    else:
        test_size = valid_size = 0
    
    # 分割数据
    train_data = selected_data[:train_size]
    if train_ratio < 1:
        test_data = selected_data[train_size:train_size + test_size]
        valid_data = selected_data[train_size + test_size:]
    
    # 保存数据
    def save_jsonl(data, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return filepath
    
    # 保存训练集
    train_file = save_jsonl(train_data, 'train.jsonl')
    console.print(f"\n[green]已保存训练集 ({len(train_data)} 条数据): {train_file}[/green]")
    
    # 保存测试集和验证集
    if train_ratio < 1:
        test_file = save_jsonl(test_data, 'test.jsonl')
        valid_file = save_jsonl(valid_data, 'valid.jsonl')
        console.print(f"[green]已保存测试集 ({len(test_data)} 条数据): {test_file}[/green]")
        console.print(f"[green]已保存验证集 ({len(valid_data)} 条数据): {valid_file}[/green]")

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
