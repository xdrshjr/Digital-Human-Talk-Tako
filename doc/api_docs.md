# DICE-Talk API 接口文档

## 概述

DICE-Talk API 提供了基于人像图片和音频生成数字人视频的RESTful接口。该API将原有的demo.py功能封装为HTTP服务，支持异步任务处理、状态查询和结果下载。

## 服务信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1
- **数据格式**: JSON
- **文档地址**: `http://localhost:8000/docs` (Swagger UI)
- **备用文档**: `http://localhost:8000/redoc` (ReDoc)

## 认证

当前版本无需认证，未来版本可能会添加API密钥或其他认证方式。

## 接口列表

### 1. 健康状态检查

检查服务运行状态和GPU可用性。

**接口地址**: `GET /api/health`

**响应示例**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "version": "1.0.0"
}
```

**响应字段说明**:
- `status`: 服务状态 ("healthy" 或 "unhealthy")
- `gpu_available`: GPU是否可用
- `model_loaded`: 模型是否加载成功
- `version`: API版本号

---

### 2. 启动数字人合成任务

提交人像图片和音频文件，启动数字人视频合成任务。

**接口地址**: `POST /api/v1/synthesis/start`

**请求方式**: `multipart/form-data`

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| image | File | 是 | 人像图片文件 (jpg, jpeg, png) |
| audio | File | 是 | 音频文件 (wav, mp3) |
| emotion | String | 是 | 情感类型 (见情感类型列表) |
| ref_scale | Float | 否 | 参考尺度 (0.0-10.0，默认3.0) |
| emo_scale | Float | 否 | 情感尺度 (0.0-10.0，默认6.0) |
| crop | Boolean | 否 | 是否裁剪图片 (默认false) |
| inference_steps | Integer | 否 | 推理步数 (8-50，默认20) |
| duration | Float | 否 | 视频时长，秒 (0.1-300.0，默认自动根据音频长度) |
| fps | Integer | 否 | 视频帧率 (8-60，默认24) |
| seed | Integer | 否 | 随机种子 (0-2147483647，默认随机) |
| dynamic_scale | Float | 否 | 动态缩放因子 (0.1-3.0，默认1.0) |
| min_resolution | Integer | 否 | 最小分辨率 (128-2048，默认256) |
| expand_ratio | Float | 否 | 扩展比例 (0.1-2.0，默认0.5) |

**情感类型列表**:
- `contempt` - 轻蔑
- `sad` - 悲伤
- `happy` - 快乐
- `surprised` - 惊讶
- `angry` - 愤怒
- `disgusted` - 厌恶
- `fear` - 恐惧
- `neutral` - 中性

**响应示例**:
```json
{
  "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "PENDING",
  "message": "Task created successfully"
}
```

**cURL示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/synthesis/start" \
  -F "image=@portrait.jpg" \
  -F "audio=@speech.wav" \
  -F "emotion=happy" \
  -F "ref_scale=3.0" \
  -F "emo_scale=6.0" \
  -F "crop=false" \
  -F "inference_steps=20" \
  -F "duration=120" \
  -F "fps=24" \
  -F "seed=12345" \
  -F "dynamic_scale=1.0" \
  -F "min_resolution=256" \
  -F "expand_ratio=0.5"
```

**文件要求**:
- **图片文件**: 
  - 格式: JPG, JPEG, PNG
  - 大小: 最大10MB
  - 要求: 包含清晰的人脸
- **音频文件**:
  - 格式: WAV, MP3
  - 大小: 最大50MB
  - 要求: 清晰的语音内容

---

### 3. 查询任务状态

根据任务ID查询合成任务的当前状态和进度。

**接口地址**: `GET /api/v1/synthesis/{task_id}/status`

**路径参数**:
- `task_id`: 任务ID (启动任务时返回)

**响应示例**:
```json
{
  "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "PROCESSING",
  "progress": 0.6,
  "message": "Generating video...",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:32:30"
}
```

**状态说明**:
- `PENDING`: 任务等待处理
- `PROCESSING`: 任务处理中
- `COMPLETED`: 任务完成
- `FAILED`: 任务失败

**进度说明**:
- `progress`: 0.0-1.0 的浮点数，表示任务完成百分比

**cURL示例**:
```bash
curl "http://localhost:8000/api/v1/synthesis/f47ac10b-58cc-4372-a567-0e02b2c3d479/status"
```

---

### 4. 下载合成视频

下载已完成的数字人视频文件。

**接口地址**: `GET /api/v1/synthesis/{task_id}/download`

**路径参数**:
- `task_id`: 任务ID (启动任务时返回)

**响应**: 
- 成功: 返回MP4视频文件
- 失败: 返回JSON错误信息

**响应头**:
```
Content-Type: video/mp4
Content-Disposition: attachment; filename="result.mp4"
```

**cURL示例**:
```bash
curl -o result.mp4 "http://localhost:8000/api/v1/synthesis/f47ac10b-58cc-4372-a567-0e02b2c3d479/download"
```

**注意事项**:
- 只有状态为 `COMPLETED` 的任务才能下载
- 视频文件为MP4格式
- 建议及时下载，服务器可能会定期清理旧文件

---

## 错误处理

### 错误响应格式

所有错误都使用统一的JSON格式返回：

```json
{
  "error": "错误信息",
  "detail": "详细错误描述（可选）"
}
```

### 常见错误码

| HTTP状态码 | 说明 |
|------------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 413 | 文件过大 |
| 422 | 请求格式错误 |
| 500 | 服务器内部错误 |

### 错误示例

**文件类型不支持**:
```json
{
  "error": "Invalid image file type. Allowed types: ['image/jpeg', 'image/png']"
}
```

**任务不存在**:
```json
{
  "error": "Task not found"
}
```

**任务未完成**:
```json
{
  "error": "Task not completed yet"
}
```

---

## 使用流程

### 完整工作流程

1. **检查服务状态**
   ```bash
   curl "http://localhost:8000/api/health"
   ```

2. **启动合成任务**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/synthesis/start" \
     -F "image=@portrait.jpg" \
     -F "audio=@speech.wav" \
     -F "emotion=happy"
   ```
   返回任务ID，例如: `f47ac10b-58cc-4372-a567-0e02b2c3d479`

3. **轮询任务状态**
   ```bash
   # 定期查询直到状态为 COMPLETED
   curl "http://localhost:8000/api/v1/synthesis/f47ac10b-58cc-4372-a567-0e02b2c3d479/status"
   ```

4. **下载结果视频**
   ```bash
   curl -o result.mp4 "http://localhost:8000/api/v1/synthesis/f47ac10b-58cc-4372-a567-0e02b2c3d479/download"
   ```

### Python客户端示例

```python
import requests
import time
import os

def synthesize_talking_portrait(image_path, audio_path, emotion="happy", ref_scale=3.0, emo_scale=6.0, 
                               inference_steps=20, duration=None, fps=24, seed=None, 
                               dynamic_scale=1.0, min_resolution=256, expand_ratio=0.5):
    base_url = "http://localhost:8000"
    
    # 1. 检查服务状态
    health = requests.get(f"{base_url}/api/health").json()
    if health["status"] != "healthy":
        raise Exception("Service not healthy")
    
    # 2. 启动任务
    with open(image_path, 'rb') as img, open(audio_path, 'rb') as aud:
        files = {
            'image': img,
            'audio': aud
        }
        data = {
            'emotion': emotion,
            'ref_scale': ref_scale,
            'emo_scale': emo_scale,
            'crop': False,
            'inference_steps': inference_steps,
            'fps': fps,
            'dynamic_scale': dynamic_scale,
            'min_resolution': min_resolution,
            'expand_ratio': expand_ratio
        }
        
        # 添加可选参数
        if duration is not None:
            data['duration'] = duration
        if seed is not None:
            data['seed'] = seed
        
        response = requests.post(f"{base_url}/api/v1/synthesis/start", files=files, data=data)
        response.raise_for_status()
        task = response.json()
        task_id = task["task_id"]
    
    print(f"Task created: {task_id}")
    
    # 3. 轮询状态
    while True:
        status_response = requests.get(f"{base_url}/api/v1/synthesis/{task_id}/status")
        status_response.raise_for_status()
        status = status_response.json()
        
        print(f"Status: {status['status']}, Progress: {status['progress']:.2%}")
        
        if status["status"] == "COMPLETED":
            break
        elif status["status"] == "FAILED":
            raise Exception(f"Task failed: {status['message']}")
        
        time.sleep(5)  # 每5秒查询一次
    
    # 4. 下载结果
    download_response = requests.get(f"{base_url}/api/v1/synthesis/{task_id}/download")
    download_response.raise_for_status()
    
    output_path = f"result_{task_id[:8]}.mp4"
    with open(output_path, 'wb') as f:
        f.write(download_response.content)
    
    print(f"Video saved to: {output_path}")
    return output_path

# 使用示例
if __name__ == "__main__":
    try:
        result = synthesize_talking_portrait(
            image_path="portrait.jpg",
            audio_path="speech.wav",
            emotion="happy",
            inference_steps=25,  # 使用更高的推理步数
            fps=30,              # 使用更高的帧率
            duration=60,         # 限制视频时长为60秒
            seed=12345           # 使用固定种子确保可重复性
        )
        print(f"Success! Video saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")

---

## 性能和限制

### 性能参数

- **并发任务数**: 最大2个同时处理的任务
- **处理时间**: 根据音频长度和参数设置，通常1分钟音频需要2-5分钟处理时间
- **GPU要求**: 建议使用NVIDIA GPU，至少8GB显存
- **CPU处理**: 支持但速度较慢，不推荐生产使用

### 资源限制

- **文件上传大小**: 图片10MB，音频50MB
- **支持的文件格式**: 
  - 图片: JPG, JPEG, PNG
  - 音频: WAV, MP3
- **任务队列**: 最多100个等待任务
- **结果保存**: 视频文件保存24小时后自动清理

### 参数建议

1. **推理步数**: 更高的步数(25-35)提高质量但增加处理时间
2. **帧率**: 24-30 FPS适合大多数用途，更高帧率增加文件大小
3. **视频时长**: 建议不超过音频长度，过短可能导致内容截断
4. **动态缩放**: 1.0为标准，1.2-1.5可增强表现力
5. **随机种子**: 固定种子确保可重复结果，适合批量处理

### 建议

1. **批量处理**: 对于大量任务，建议分批提交避免队列堆积
2. **文件优化**: 上传前压缩图片和音频可提高处理速度
3. **错误重试**: 网络不稳定时建议添加重试机制
4. **结果下载**: 及时下载处理结果，避免文件被自动清理

---

## 部署和运维

### 启动服务

```bash
# 一键启动（推荐）
./start_server.sh

# 或者手动启动
python -m api.server
```

### 服务监控

```bash
# 检查服务状态
curl http://localhost:8000/api/health

# 查看服务日志
tail -f logs/api.log

# 检查GPU使用情况
nvidia-smi
```

### 常见问题

1. **GPU内存不足**: 减少并发任务数或使用更大显存的GPU
2. **模型加载失败**: 检查模型文件是否完整下载
3. **端口被占用**: 修改配置文件中的端口设置
4. **依赖包缺失**: 运行 `pip install -r requirements_api.txt`

---

## 更新日志

### v1.1.0 (2024-01-16)
- **新增用户参数支持**: 支持推理步数、视频时长、帧率等用户自定义参数
- **增强参数控制**: 用户可精确控制视频生成质量和特性
- **改进缓存机制**: 基于完整参数集的智能缓存
- **扩展API文档**: 详细说明所有可用参数及其用途

### v1.0.0 (2024-01-15)
- 初始版本发布
- 支持基本的数字人视频合成
- 提供完整的RESTful API
- 支持异步任务处理
- 添加健康检查和状态监控

---

## 联系支持

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [DICE-Talk项目](https://github.com/toto222/DICE-Talk)
- 技术文档: [项目主页](https://toto222.github.io/DICE-Talk/)

---

*本文档最后更新时间: 2024-01-16*