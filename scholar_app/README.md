# scholar_app

CLI 与浏览器 demo 入口。这里不放检索业务逻辑，只负责接收命令行参数、启动 Web server，并调用 `cost_control_cache.SearchPipeline`。

常用命令：

```bash
python -m scholar_app.cli --backend auto search --query "image retrieval" --top-k 5
python -m scholar_app.cli --backend database search --query "image retrieval" --top-k 3
python -m scholar_app.cli --backend auto serve --port 8765
```
