import wandb
import time
import os
from dotenv import load_dotenv
import subprocess
from rich.console import Console

# 加载环境变量
load_dotenv()
console = Console()

def test_wandb_logging():
    """测试 wandb 记录功能"""
    # 获取 wandb API key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        console.print("[red]错误: 未找到 WANDB_API_KEY[/red]")
        return
    
    # 登录 wandb
    try:
        wandb.login(key=wandb_api_key)
        console.print("[green]成功登录 wandb[/green]")
    except Exception as e:
        console.print(f"[red]wandb 登录失败: {str(e)}[/red]")
        return
    
    # 初始化 wandb 运行
    run = wandb.init(
        project="mlx-finetune-test",
        name="debug_run",
        config={
            "model": "test_model",
            "batch_size": 1,
            "learning_rate": 0.001
        }
    )
    
    if wandb.run is None:
        console.print("[red]wandb run 初始化失败[/red]")
        return
    else:
        console.print(f"[green]wandb run 初始化成功: {wandb.run.name}[/green]")
    
    # 定义要追踪的指标
    wandb.define_metric("train/global_step", summary="max")
    wandb.define_metric("train/epoch", summary="max")
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/grad_norm", summary="mean")
    wandb.define_metric("train/learning_rate", summary="last")
    
    try:
        # 模拟训练过程
        total_steps = 100
        for step in range(total_steps):
            # 模拟训练指标
            metrics = {
                "train/global_step": step,
                "train/epoch": step / total_steps,
                "train/loss": 2.0 * (1 - step/total_steps),  # 损失随步数减小
                "train/grad_norm": 0.1 * (1 - step/total_steps),  # 梯度范数随步数减小
                "train/learning_rate": 0.001 * (1 - step/total_steps)  # 学习率随步数减小
            }
            
            # 记录指标
            wandb.log(metrics, step=step)
            console.print(f"Step {step}: {metrics}")
            
            # 暂停一下，模拟训练间隔
            time.sleep(0.1)
            
    except Exception as e:
        console.print(f"[red]记录指标时出错: {str(e)}[/red]")
    finally:
        # 完成记录
        wandb.finish()
        console.print("[green]测试完成[/green]")

if __name__ == "__main__":
    test_wandb_logging() 