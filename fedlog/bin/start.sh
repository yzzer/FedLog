#!/bin/bash

# 默认值
APP_MODULE="main"
HOST="127.0.0.1"
PORT="8000"
RELOAD="false"

# 帮助信息
show_help() {
    echo "用法: ./start_uvicorn.sh [-a APP_MODULE] [-h HOST] [-p PORT] [-r]"
    echo ""
    echo "参数:"
    echo "  -a    指定要运行的 FastAPI app (格式: 模块名:应用实例, 默认为 main:app)"
    echo "  -h    指定服务器的主机地址 (默认为 127.0.0.1)"
    echo "  -p    指定服务器的端口号 (默认为 8000)"
    echo "  -r    启用自动重载 (适用于开发环境)"
    exit 0
}

# 解析输入参数
while getopts "a:h:p:rh" opt; do
  case ${opt} in
    a )
      APP_MODULE=$OPTARG
      ;;
    h )
      HOST=$OPTARG
      ;;
    p )
      PORT=$OPTARG
      ;;
    r )
      RELOAD="true"
      ;;
    ? )
      show_help
      ;;
  esac
done

# 启动 Uvicorn
if [ "$RELOAD" == "true" ]; then
  uvicorn server.${APP_MODULE}:app --host $HOST --port $PORT --reload
else
  uvicorn server.${APP_MODULE}:app --host $HOST --port $PORT
fi
