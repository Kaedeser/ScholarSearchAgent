# Query Intent Service NodePort Deployment

## Current Service

The query intent model is deployed on `cu05` with Kubernetes namespace
`default`.

```text
Remote app root: /data/csp/query-intent-service
Kubernetes deployment: query-intent-service
Kubernetes service: query-intent-service
Service type: NodePort
Container port: 8080
NodePort: 22436
External test URL: http://10.99.24.182:22436
```

The container uses its own image environment:

```text
ai-harbor.wust.edu.cn/master-23-liuhaijun/bert:1
```

The container does not mount or call the host virtual environment. The only
hostPath mount is:

```text
/data/csp/query-intent-service -> /app
```

For host-side manual debugging outside Kubernetes, use the host virtual
environment:

```bash
/data/k8s/anaconda3/envs/py-train/bin/python
```

## Remote Files

```text
/data/csp/query-intent-service
  serve_query_intent.py
  models/query_gate_biobert
  models/intent_biobert
  k8s/query-intent-service.yaml
```

Local source files:

```text
deploy/serve_query_intent.py
deploy/query-intent-service.yaml
outputs/query_gate_biobert
outputs/intent_biobert
```

## API

Health check:

```bash
curl http://10.99.24.182:22436/health
```

Prediction:

```bash
curl -X POST http://10.99.24.182:22436/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "auto",
    "texts": [
      "Could you recommend recent papers about neural information retrieval?",
      "Write a Python function to sort a list."
    ]
  }'
```

`mode` can be:

```text
auto
gate
intent
```

In `auto` mode, the service first runs the gate model. If the gate label is
`paper_search`, the service then runs the intent model. Non-paper queries return
`intent: null`.

## Kubernetes Commands

Apply or re-apply:

```bash
kubectl apply -f /data/csp/query-intent-service/k8s/query-intent-service.yaml
```

Check status:

```bash
kubectl get deploy,pod,svc -n default -l app=query-intent-service -o wide
kubectl rollout status deployment/query-intent-service -n default
```

View logs:

```bash
POD=$(kubectl get pod -n default -l app=query-intent-service -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n default "$POD" --tail=100
```

Delete deployment:

```bash
kubectl delete -f /data/csp/query-intent-service/k8s/query-intent-service.yaml
```

## Verification Result

Verified from outside the cluster:

```text
GET  http://10.99.24.182:22436/health
POST http://10.99.24.182:22436/predict
```

Expected sample labels:

```text
"Could you recommend recent papers about neural information retrieval?"
  gate: paper_search
  intent: method_search

"Write a Python function to sort a list."
  gate: non_paper_search
  intent: null
```
