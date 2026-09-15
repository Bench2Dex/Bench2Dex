import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RELATIVIZE_SCRIPT = SCRIPT_DIR / "relativize_usd_asset_paths.py"


def _extract_output_usd_path(cmd: str) -> str | None:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None

    for token in tokens:
        if token.lower().endswith((".usd", ".usda", ".usdc")):
            return os.path.abspath(token)
    return None


def _relativize_usd_paths(output_usd_path: str) -> None:
    cmd = [sys.executable, str(RELATIVIZE_SCRIPT), output_usd_path]
    subprocess.run(cmd, check=True)


def _command_handles_relativize_internally(cmd: str) -> bool:
    normalized = cmd.replace("\\", "/")
    return "relativize_usd_asset_paths.py" in normalized or "convert_urdf_sdf_instance.py" in normalized

def main():
    txt_path = 'convert.txt'
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # 处理一行内涉及分号的多个操作 (如 077_rubiks_cube 包含了3并联指令)
        sub_commands = [cmd.strip() for cmd in line.split(';')]
        
        for cmd_idx, cmd in enumerate(sub_commands):
            if not cmd:
                continue
                
            # 自动追加无头参数提高渲染稳定性和降低内存开销
            if '--headless' not in cmd:
                cmd = cmd + " --headless"
                
            print(f"\n{'='*80}")
            print(f"[ 行进度 {line_idx+1}/{total_lines} | 子命令进度 {cmd_idx+1}/{len(sub_commands)} ]")
            print(f"执行命令: {cmd}")
            print(f"{'='*80}\n")
            
            # 串行阻塞：等这个进程完全结束 (Shutting Down 之后) 才往下走
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[错误] 命令执行异常终止，退出码: {e.returncode}")
                # 为了防止一个坏掉全盘崩溃，遇到报错不直接退出程序，而是继续下一个任务
                continue

            if _command_handles_relativize_internally(cmd):
                continue

            output_usd_path = _extract_output_usd_path(cmd)
            if output_usd_path is None:
                print("[警告] 未能从命令中识别输出 USD，跳过相对路径后处理。")
                continue
            if not os.path.exists(output_usd_path):
                print(f"[警告] 输出 USD 不存在，跳过相对路径后处理: {output_usd_path}")
                continue

            try:
                _relativize_usd_paths(output_usd_path)
            except subprocess.CalledProcessError as e:
                print(f"\n[错误] USD 相对路径后处理失败，退出码: {e.returncode}")
                continue

if __name__ == '__main__':
    main()
