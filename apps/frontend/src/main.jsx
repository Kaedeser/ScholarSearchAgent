// 中文功能说明：React 应用入口，负责渲染论文检索工作台根组件。

import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import "./styles/base.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
