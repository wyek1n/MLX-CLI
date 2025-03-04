import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich import print
from typing import Optional
import os
import subprocess
from enum import Enum

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
    except ImportError:
        console.print("[red]✗[/red] 请先安装 MLX 框架")
        raise typer.Exit(1)

def show_menu():
    """显示主菜单"""
    console.clear()  # 清屏
    console.print("\n[bold cyan]请选择操作:[/bold cyan]")
    console.print("[1] 模型下载")
    console.print("[2] 数据准备")
    console.print("[3] 模型微调")
    console.print("[4] 模型合并")
    console.print("[5] 模型评估")
    console.print("[0] 退出程序")
    
    choice = IntPrompt.ask("\n请输入选项", choices=["0", "1", "2", "3", "4", "5"])
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

if __name__ == "__main__":
    app() 