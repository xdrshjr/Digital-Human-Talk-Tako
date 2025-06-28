# DICE-Talk API 服务

这是为DICE-Talk项目添加的FastAPI服务器，将原有的demo.py功能封装为RESTful API接口，支持远程调用和前后端分离架构。

## 快速开始

### 1. 环境要求

- Ubuntu 18.04+ (推荐 Ubuntu 20.04)
- Python 3.8+
- NVIDIA GPU (推荐8GB+显存)
- CUDA 11.0+

### 2. 一键启动

在Ubuntu系统上，使用以下命令一键启动服务：

```bash
# 给启动脚本执行权限
chmod +x start_server.sh

# 启动服务
./start_server.sh
```

启动脚本会自动：
- 检查系统环境
- 安装Python依赖
- 创建必要目录
- 启动API服务器
- 执行健康检查

### 3. 服务地址

服务启动后，访问以下地址：

- **API基础地址**: http://localhost:8000
- **健康检查**: http://localhost:8000/api/health
- **API文档**: http://localhost:8000/docs (Swagger UI)
- **备用文档**: http://localhost:8000/redoc (ReDoc)

## 功能特性

### ✅ 核心功能
- **数字人视频合成**: 基于人像图片和音频生成数字人视频
- **异步任务处理**: 支持长时间任务的后台处理
- **任务状态查询**: 实时查询合成进度和状态
- **结果文件下载**: 安全的文件下载接口
- **健康状态监控**: 服务和GPU状态检查

### ✅ 技术特性
- **RESTful API**: 标准的REST接口设计
- **文件验证**: 严格的文件类型和大小验证
- **错误处理**: 完善的错误信息和异常处理
- **并发控制**: 智能的任务队列管理
- **安全机制**: 路径遍历攻击防护

## API接口

### 1. 健康检查
```http
GET /api/health
```

### 2. 启动合成任务
```http
POST /api/v1/synthesis/start
Content-Type: multipart/form-data

参数:
- image: 人像图片文件 (jpg/png, <10MB)
- audio: 音频文件 (wav/mp3, <50MB)  
- emotion: 情感类型 (happy/sad/angry等)
- ref_scale: 参考尺度 (0.0-10.0, 默认3.0)
- emo_scale: 情感尺度 (0.0-10.0, 默认6.0)
- crop: 是否裁剪 (true/false, 默认false)
```

### 3. 查询任务状态
```http
GET /api/v1/synthesis/{task_id}/status
```

### 4. 下载结果视频
```http
GET /api/v1/synthesis/{task_id}/download
```

详细的API文档请查看: [doc/api_docs.md](doc/api_docs.md)

## 使用示例

### cURL示例

```bash
# 1. 健康检查
curl http://localhost:8000/api/health

# 2. 启动任务
curl -X POST "http://localhost:8000/api/v1/synthesis/start" \
  -F "image=@examples/img/nazha.png" \
  -F "audio=@examples/wav/female-zh.wav" \
  -F "emotion=happy"

# 3. 查询状态 (使用返回的task_id)
curl http://localhost:8000/api/v1/synthesis/{task_id}/status

# 4. 下载结果
curl -o result.mp4 http://localhost:8000/api/v1/synthesis/{task_id}/download
```

### Python客户端示例

```python
import requests

# 启动合成任务
with open('portrait.jpg', 'rb') as img, open('speech.wav', 'rb') as aud:
    files = {'image': img, 'audio': aud}
    data = {'emotion': 'happy', 'ref_scale': 3.0, 'emo_scale': 6.0}
    
    response = requests.post(
        'http://localhost:8000/api/v1/synthesis/start',
        files=files, data=data
    )
    task = response.json()
    task_id = task['task_id']

# 查询状态
status = requests.get(f'http://localhost:8000/api/v1/synthesis/{task_id}/status')
print(status.json())

# 下载结果
if status.json()['status'] == 'COMPLETED':
    result = requests.get(f'http://localhost:8000/api/v1/synthesis/{task_id}/download')
    with open('result.mp4', 'wb') as f:
        f.write(result.content)
```

## 测试验证

项目包含测试脚本用于验证API功能：

```bash
# 快速健康检查
python test_api.py health

# 完整功能测试
python test_api.py
```

测试脚本会验证：
- 服务健康状态
- 任务启动功能
- 状态查询功能
- 结果下载功能

## 部署配置

### 配置文件

服务配置文件位于 `config/api_config.yaml`，包含：

- 服务器设置 (端口、主机等)
- GPU设置 (设备ID等)
- 任务设置 (并发数等)
- 文件设置 (大小限制等)

### 日志文件

服务日志保存在 `logs/api.log`，包含：

- 服务启动/关闭日志
- 任务处理日志
- 错误异常日志

### 目录结构

```
DICE-Talk/
├── api/                 # API服务代码
│   ├── __init__.py
│   ├── server.py        # FastAPI主服务器
│   ├── models.py        # 数据模型
│   ├── task_manager.py  # 任务管理器
│   └── utils.py         # 工具函数
├── config/
│   └── api_config.yaml  # API配置文件
├── doc/
│   └── api_docs.md      # 详细API文档
├── uploads/             # 上传文件目录
├── outputs/             # 输出文件目录
├── logs/                # 日志目录
├── start_server.sh      # 启动脚本
├── test_api.py          # 测试脚本
├── requirements_api.txt # API依赖
└── README_API.md        # 本文档
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 杀死占用进程
   kill -9 <PID>
   ```

2. **GPU内存不足**
   ```bash
   # 检查GPU使用情况
   nvidia-smi
   
   # 减少并发任务数
   # 编辑 config/api_config.yaml 中的 max_workers
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件是否存在
   ls -la checkpoints/
   
   # 重新下载模型文件
   ```

4. **依赖包问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt
   pip install -r requirements_api.txt
   ```

### 日志查看

```bash
# 查看实时日志
tail -f logs/api.log

# 查看错误日志
grep ERROR logs/api.log

# 查看任务日志
grep "Task" logs/api.log
```

## 性能优化

### 硬件建议

- **GPU**: NVIDIA RTX 3080/4080 或更高
- **内存**: 32GB+ 系统内存
- **存储**: SSD硬盘，至少100GB可用空间

### 软件优化

- 调整并发任务数 (`max_workers`)
- 优化推理步数 (`inference_steps`)
- 使用更小的分辨率 (`min_resolution`)
- 启用GPU混合精度

## 开发扩展

### 添加新接口

1. 在 `api/models.py` 中定义数据模型
2. 在 `api/server.py` 中实现接口逻辑
3. 更新 `doc/api_docs.md` 文档
4. 在 `test_api.py` 中添加测试

### 集成前端

API遵循标准RESTful设计，可轻松集成：

- Vue.js/React前端
- 移动端应用
- 桌面客户端
- 其他后端服务

## 许可证

本API服务遵循与DICE-Talk项目相同的许可证，仅供非商业用途。

## 联系支持

- **项目主页**: https://toto222.github.io/DICE-Talk/
- **GitHub**: https://github.com/toto222/DICE-Talk
- **问题反馈**: GitHub Issues

---

*最后更新: 2024-01-15* 