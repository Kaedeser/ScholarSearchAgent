# Selector Reranker Service Deployment

Remote layout on cu05:

```text
/data/csp/selector-reranker/
  model/      # HuggingFace/CrossEncoder model files
  service/    # app.py and Dockerfile
  k8s/        # Kubernetes manifests
```

Current running model verified on 2026-07-04:

```text
Deployment: selector-reranker
Pod: selector-reranker-68646cfd46-jw6ng
Node: 11.11.11.5
Host model path: /data/csp/selector-reranker/model
Container MODEL_DIR: /models/selector-reranker
Threshold: 0.0006931035313755274
Model files mtime: 2026-07-03 16:18
Dev accuracy: 0.8813559322033898
Dev F1: 0.8880994671403197
Average precision: 0.9638932054125722
```

Build image on cu05:

```bash
cd /data/csp/selector-reranker/service
docker build -t selector-reranker-service:20260624 .
```

Deploy to Kubernetes default namespace:

```bash
kubectl apply -f /data/csp/selector-reranker/k8s/selector-reranker.yaml
kubectl rollout status deployment/selector-reranker -n default --timeout=600s
```

NodePort:

```text
http://10.99.24.182:32082
```

Test:

```bash
curl http://10.99.24.182:32082/health
curl -X POST http://10.99.24.182:32082/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which papers discuss video captioning?","top_k":2,"documents":[{"id":"p1","title":"Video Captioning with Transformers","abstract":"A method for generating natural language descriptions for videos."},{"id":"p2","title":"Database Indexing","abstract":"An index structure for transaction processing."}]}'
```
