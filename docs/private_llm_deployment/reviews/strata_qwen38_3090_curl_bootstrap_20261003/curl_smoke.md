# curl smoke — 2026-10-04

Endpoint: `http://127.0.0.1:8081`; bound to loopback only. Exactly one health request, one models request, and one chat completion were issued.

## Health

`curl -sS -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8081/health`

HTTP 200; response fields: `status=ok`, `max_context=131072`, `model=qwen3.8-flash-next-iq3_xxs`, `images=false`, `loaded=true`, `service=strata`.

## Models

`curl -sS -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8081/v1/models`

HTTP 200; model id `qwen3.8-flash-next-iq3_xxs`, status `loaded`, `n_ctx=131072`, text input/output modalities.

## Chinese chat completion

One `POST /v1/chat/completions`, using the returned model id, `max_tokens=256`, `temperature=0`, completed with HTTP 200 in 4.382 seconds. Usage: 68 prompt tokens, 167 completion tokens, 235 total. The returned answer was three Chinese sentences:

> 大型化工工程投标技术文件应重点响应招标技术要求，明确工艺路线、设备选型、设计标准及关键技术指标。
> 同时需体现施工组织能力，包括进度计划、质量控制、HSE管理、安装调试及试运行方案。
> 还应突出企业资质、类似项目业绩、人员配置、风险识别与应对措施，以证明履约能力。
