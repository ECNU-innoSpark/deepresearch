#!/usr/bin/env bash
# Deep Research 一键环境创建 + RAGFlow Docker 启动
# 用法: ./setup.sh  或  bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
RAGFLOW_DOCKER="$PROJECT_ROOT/ragflow/docker"

echo "=============================================="
echo "  Deep Research 环境配置"
echo "=============================================="
echo "  项目根目录: $PROJECT_ROOT"
echo "  Deep Research: $SCRIPT_DIR"
echo "=============================================="

# 1. 检查 Python 3.10+
if ! command -v python3 &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "[错误] 未找到 Python，请先安装 Python 3.10+"
        exit 1
    fi
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")' 2>/dev/null || echo "0")
if [[ "$(echo "$PYTHON_VERSION" | cut -d. -f1)" -lt 3 ]] || [[ "$(echo "$PYTHON_VERSION" | cut -d. -f2)" -lt 10 ]]; then
    echo "[错误] 需要 Python 3.10+，当前: $($PYTHON_CMD --version 2>&1)"
    exit 1
fi
echo "[OK] Python: $($PYTHON_CMD --version)"

# 2. 创建并激活虚拟环境
if [[ -d "$VENV_DIR" ]]; then
    echo "[跳过] 虚拟环境已存在: $VENV_DIR"
else
    echo "[1/4] 创建虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "[OK] 虚拟环境已创建: $VENV_DIR"
fi

# 激活虚拟环境 (当前 shell)
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # Linux/macOS
    source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Windows Git Bash
    source "$VENV_DIR/Scripts/activate"
else
    echo "[错误] 未找到 activate 脚本"
    exit 1
fi

# 3. 安装依赖
echo "[2/4] 安装 Deep Research 依赖..."
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "[OK] 依赖安装完成"

# 4. 环境变量文件
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        echo "[OK] 已从 .env.example 创建 .env，请按需修改 API Key"
    fi
else
    echo "[跳过] .env 已存在"
fi

# 5. 启动 RAGFlow Docker（可选）
echo "[3/4] 检查 RAGFlow Docker..."
if [[ -d "$RAGFLOW_DOCKER" ]] && [[ -f "$RAGFLOW_DOCKER/docker-compose.yml" ]]; then
    if command -v docker &>/dev/null; then
        echo "[4/4] 启动 RAGFlow 容器 (CPU 模式)..."
        (
            cd "$RAGFLOW_DOCKER"
            docker compose -f docker-compose.yml --profile cpu up -d
        )
        echo "[OK] RAGFlow 已在后台启动"
        echo "     Web UI: http://localhost:80   API: http://localhost:9380"
        echo "     查看日志: docker logs -f \$(docker ps -q -f name=ragflow)"
    else
        echo "[跳过] 未检测到 docker 命令，跳过 RAGFlow 启动"
    fi
else
    echo "[跳过] 未找到 ragflow/docker 目录，跳过 RAGFlow 启动"
fi

echo ""
echo "=============================================="
echo "  配置完成"
echo "=============================================="
echo "  激活环境: source $VENV_DIR/bin/activate"
echo "  运行示例: python main.py \"你的研究问题\" --mock -v"
echo "  正式运行: python main.py \"你的研究问题\" -v"
echo "=============================================="
