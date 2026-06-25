# apps/frontend

中文功能说明：ScholarSearchAgent 的独立 React 前端工程，提供正式论文检索工作台页面，只通过 HTTP 调用后端公开 API。

## 技术栈

- React 19
- Vite 6
- lucide-react 图标
- 原生 fetch 调用后端 `/health` 和 `/api/search`

## 安装依赖

从前端目录运行：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent\apps\frontend
npm install
```

## 开发启动

```powershell
npm run dev
```

默认访问地址：

```text
http://127.0.0.1:5174
```

前端默认后端 API：

```text
http://127.0.0.1:8765
```

页面顶部的 API 输入框可以切换后端地址，并会保存到浏览器 `localStorage`。

## 生产构建

```powershell
npm run build
```

构建产物输出到：

```text
apps/frontend/dist
```

本地预览生产构建：

```powershell
npm run preview
```

## 目录结构

```text
src/
  api/          后端 API 客户端
  components/   页面通用组件
  pages/        页面级容器
  styles/       全局样式系统
  utils/        展示格式化工具
  App.jsx       React 应用根组件
  main.jsx      Vite 入口
```

## 后端联调

后端单独启动：

```powershell
cd F:\中国研究生人工智能大赛\ScholarSearchAgent
python -m apps.backend.scholar_api.cli --backend auto serve --host 127.0.0.1 --port 8765
```

后端 API：

```text
GET /health
GET /api/search?q=<query>&top_k=<number>
```
