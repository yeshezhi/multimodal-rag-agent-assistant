# Docker Compose、Milvus 与 Nginx 部署

## 组件与端口

- `app`：FastAPI，容器内端口 `8000`。
- `milvus`：文本向量数据库，仅在 Compose 网络内开放。
- `nginx`：反向代理，仅绑定服务器 `127.0.0.1:8011`。
- `etcd`、`minio`：Milvus 的内部依赖，不对宿主机公开。

因此 Windows 端需要新建隧道：

```powershell
ssh -N -L 8011:127.0.0.1:8011 my-ai-server
```

然后访问 `http://127.0.0.1:8011/`。

## 首次部署

```bash
cd /home/cjy/project/multimodal-rag-assistant
cp .env.docker.example .env.docker
docker compose --env-file .env.docker up -d etcd minio milvus
docker compose --env-file .env.docker --profile tools run --rm migrate
docker compose --env-file .env.docker up -d --build app nginx
docker compose --env-file .env.docker ps
curl http://127.0.0.1:8011/health
```

迁移步骤从已有 `data/faiss.index` 与 `data/chunks.json` 读取向量和元数据，写入 Milvus。迁移命令带 `--replace`，会覆盖同名 Milvus collection，但不会删除原 FAISS 文件；在确认 Milvus 正常前保留 FAISS 作为回退。

## GPU 模式

默认 Compose 配置使用 CPU，保证在没有 NVIDIA Container Toolkit 的服务器上也能启动。若要让容器使用 4090，请先由服务器管理员安装并验证 NVIDIA Container Toolkit：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

确认成功后，以 GPU 覆盖文件启动：

```bash
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml up -d --build app nginx
```

## 常用运维命令

```bash
docker compose --env-file .env.docker logs -f app
docker compose --env-file .env.docker logs -f milvus
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker down
```

`docker compose --env-file .env.docker down` 不会删除 named volumes；只有显式执行 `docker compose --env-file .env.docker down -v` 才会删除 Milvus、etcd 与 MinIO 的持久化数据。
