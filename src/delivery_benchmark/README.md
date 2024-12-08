# Benchmark使用说明

## 安装依赖

```sh
apt install psmisc
```

## 生成配送任务指令

```sh
python3 delivery_request_generator.py --num 10 --per-block 2 --output-file data/xxx.txt
```

## 自动化执行配送任务指令

```sh
./start_delivery.sh data/xxx.txt
```