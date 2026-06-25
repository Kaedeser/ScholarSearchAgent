# scholar_frontend

ScholarSearchAgent 的独立静态前端。

该目录不依赖 npm、Vite 或打包工具，直接用任意静态文件服务器启动即可。

## 启动

从项目根目录运行：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent
python -m http.server 5173 --directory scholar_frontend
```

浏览器打开：

```text
http://127.0.0.1:5173
```

默认后端 API 地址：

```text
http://127.0.0.1:8765
```

页面顶部的 API 输入框可以改成其他后端地址，并会保存到浏览器 localStorage。

## 后端

后端单独启动：

```powershell
python -m scholar_app.cli --backend auto serve --host 127.0.0.1 --port 8765
```

后端只提供 JSON API：

```text
GET /health
GET /api/search?q=<query>&top_k=<number>
```
