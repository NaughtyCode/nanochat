"""
nanochat 通用工具模块。
提供计算 dtype 检测、DDP 初始化、日志系统、GPU FLOPs 表等功能。
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# 计算使用的 dtype（矩阵乘法、激活）。主权重保持 fp32 以保证优化器精度。
# Linear 层在 forward 时将权重转换为此 dtype，替代 torch.amp.autocast。
# 可通过环境变量 NANOCHAT_DTYPE 覆盖: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    """自动检测最优计算 dtype：
    - bf16 需要 SM 80+ (Ampere: A100, A10 等)
    - 更旧的 GPU (V100 SM 70, T4 SM 75) 只有 fp16 张量核心
    - fp16 训练需要 GradScaler（尚未实现），因此回退到 fp32
    """
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器：添加 ANSI 颜色以增强可读性。"""
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 品红
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            # 高亮数字、百分比和分片号
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    """设置带颜色的默认日志系统"""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    """获取 nanochat 的基础目录。默认为 ~/.cache/nanochat。
    可通过环境变量 NANOCHAT_BASE_DIR 覆盖。"""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """带文件锁的下载函数。
    使用文件锁防止多 rank 并发下载同一文件。
    postprocess_fn: 可选的后处理函数，下载完成后调用。"""
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # 只有一个 rank 能获取此锁，其他 rank 阻塞等待
        # 获取锁后重新检查（可能其他 rank 已下载完成）
        if os.path.exists(file_path):
            return file_path

        # 下载文件内容
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        # 写入本地文件
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # 如果有后处理函数，执行它
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    """仅在 rank 0 时打印，避免多 GPU 训练时的重复日志。"""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    """打印 nanochat ASCII 艺术 Banner（DOS Rebel 字体）"""
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """检测是否通过 torchrun 启动（环境变量存在）。
    用于决定是否应该初始化进程组。"""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """检测 torch.distributed 是否已初始化。
    在清理时使用，避免销毁不存在的进程组。"""
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    """获取分布式训练信息 (is_ddp, rank, local_rank, world_size)。
    依赖 torchrun 设置的环境变量。"""
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    """自动检测设备类型：优先级 CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"):
    """基础计算初始化（设备、精度、DDP）。
    统一的初始化入口，包括：
    - 可复现性（设置全局种子）
    - 精度设置（TF32 用于 CUDA）
    - 分布式初始化（DDP，可选，需要 CUDA）
    返回 (is_ddp, rank, local_rank, world_size, device)
    """

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # 设置全局种子保证可复现性（大部分代码使用显式 rng 对象）
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    # CUDA 精度：启用 TF32 用于矩阵乘法
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # 分布式设置：DDP 需要 CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """compute_init 的配套清理函数，在脚本退出前调用"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """当不使用 wandb 时的虚拟替代品，保持相同的 API 签名"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# 各种 GPU 的 BF16 峰值 FLOPs 硬编码表
# 参考 torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
def get_peak_flops(device_name: str) -> float:
    """根据 GPU 设备名称返回 BF16 峰值 FLOPs。
    用于计算 MFU（Model FLOPs Utilization，模型 FLOPs 利用率）。
    未知 GPU 返回 inf，使 MFU 显示为 0%（而非错误的猜测值）。
    表顺序很重要：更具体的模式匹配优先。"""
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere 数据中心
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada 数据中心
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA 加速器
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # 消费级 RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Intel Ponte Vecchio (PVC) - 根据计算单元数量动态计算
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # 未知 GPU - 返回 inf 使 MFU 显示为 0%
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
