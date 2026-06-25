// 中文功能说明：React 前端 API 基础客户端，负责保存后端地址、拼接 URL 和统一处理 JSON 响应。

export const DEFAULT_API_BASE = "http://127.0.0.1:8765";
export const API_BASE_KEY = "scholarSearchApiBase";

export function loadApiBase() {
  return localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE;
}

export function saveApiBase(value) {
  const cleaned = cleanApiBase(value);
  localStorage.setItem(API_BASE_KEY, cleaned);
  return cleaned;
}

export function cleanApiBase(value) {
  return String(value || DEFAULT_API_BASE).trim().replace(/\/+$/, "");
}

export async function requestJson(baseUrl, path, options = {}) {
  const response = await fetch(`${cleanApiBase(baseUrl)}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  }
  return data;
}
