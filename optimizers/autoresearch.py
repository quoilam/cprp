import os
import shutil
import subprocess
import platform


def _get_data_dir() -> str:
    """Return the absolute path to the data directory bundled with this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def start_experiment(
    train_file_path: str,
    scenario_description: str,
    user_image_file_path: str | None = None,
    prepare_file_path: str | None = None,
) -> str | None:
    data_dir = _get_data_dir()
    program_md_path = os.path.join(data_dir, "program.md")
    backup_md_path = os.path.join(data_dir, "program.md.bak")

    if not os.path.exists(program_md_path):
        print("Error: 未找到 program.md 文件！")
        return None

    with open(program_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 备份原始文件，防止占位符被永久替换
    shutil.copy2(program_md_path, backup_md_path)

    # 替换占位符
    content = content.replace("{{TRAIN_FILE_PATH}}", train_file_path)
    content = content.replace(
        "{{USER_IMAGE_FILE_PATH}}",
        user_image_file_path if user_image_file_path else "User Image File Path",
    )
    content = content.replace("{{SCENARIO_DESCRIPTION}}", scenario_description)
    resolved_prepare_file_path = prepare_file_path or os.path.join(
        data_dir, "prepare.py")
    content = content.replace(
        "{{PREPARE_FILE_PATH}}", resolved_prepare_file_path.replace("\\", "\\\\"))

    if user_image_file_path:
        if "User Image File Path:" not in content:
            content += f"\n# The user image file path\nUser Image File Path:{user_image_file_path}\n"

    with open(program_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        # 构造并启动命令行，根据操作系统选择不同的执行脚本
        if platform.system() == "Windows":
            script_path = os.path.join(data_dir, "run_exp.ps1")
            command = f'powershell -ExecutionPolicy Bypass -File "{script_path}"'
        else:
            script_path = os.path.join(data_dir, "run_exp.sh")
            if os.path.exists(script_path):
                os.chmod(script_path, 0o755)
            command = f'bash "{script_path}"'

        print(f"正在启动实验，执行命令: {command}")
        subprocess.run(command, shell=True, cwd=data_dir)
    finally:
        # 无论实验成功与否，都恢复原始 program.md
        if os.path.exists(backup_md_path):
            shutil.move(backup_md_path, program_md_path)
            print("已恢复 program.md 原始内容。")

    return train_file_path


def optimize(
    algo_file_path: str,
    user_prompt: str,
    user_image_file_path: str,
    prepare_file_path: str | None = None,
) -> None:
    """
    Adapter function for cprp pipeline.

    Matches the OptimizerAdapter signature in cprp/pipeline/stages.py:
        optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None
    """
    start_experiment(algo_file_path, user_prompt,
                     user_image_file_path, prepare_file_path)
