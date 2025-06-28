# DICE-Talk FastAPI Backend Service

**A Digital Human FastAPI Backend Service Powered by DICE-Talk Technology**

<a href='https://toto222.github.io/DICE-Talk/'><img src='https://img.shields.io/badge/Original-Project-Green'></a>
<a href='https://arxiv.org/abs/2504.18087'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://github.com/toto222/DICE-Talk" style="margin: 0 2px;">
  <img src='https://img.shields.io/badge/Source-DICE--Talk-blue?style=flat&logo=github' alt='Original Repository'>
</a>
<a href="https://raw.githubusercontent.com/toto222/DICE-Talk/refs/heads/main/LICENSE" style="margin: 0 2px;">
  <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
</a>

## 🚀 Overview

This project provides a **RESTful API backend service** for digital human talking portrait generation, built upon the groundbreaking **DICE-Talk** technology. The core technology is entirely based on the original [DICE-Talk project](https://github.com/toto222/DICE-Talk) by Tan et al., which introduces "Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation."

**Core Technology Source**: This service leverages the complete DICE-Talk framework, including:
- Identity-preserving talking portrait generation
- Emotion-aware facial animation
- Audio-driven lip synchronization
- High-quality video synthesis using Stable Video Diffusion

## 🎯 Key Features

### 🎭 Digital Human Capabilities
- **Portrait Animation**: Generate realistic talking videos from static images
- **Emotion Control**: Support for multiple emotions (happy, sad, angry, surprised, neutral)
- **Identity Preservation**: Maintain original person's identity while adding emotions
- **Audio Synchronization**: Perfect lip-sync with input audio

### 🛠️ API Service Features
- **RESTful API**: Standard HTTP API for easy integration
- **Asynchronous Processing**: Non-blocking task handling for long-running operations
- **Real-time Status Tracking**: Monitor synthesis progress and status
- **Secure File Handling**: Validated file upload and secure download
- **Health Monitoring**: Service and GPU status monitoring

## 🔑 Quick Start

### Prerequisites
- Ubuntu 18.04+ (recommended Ubuntu 20.04)
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0+

### 1. One-Click Setup

```bash
# Grant execution permission
chmod +x start_server.sh

# Start the service
./start_server.sh
```

The startup script will automatically:
- Check system environment
- Install Python dependencies
- Create necessary directories
- Start the API server
- Perform health checks

### 2. Service Endpoints

Once started, access the following URLs:

- **API Base URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

## 📋 API Usage

### Basic Workflow

1. **Health Check**: Verify service status
2. **Start Task**: Upload image, audio, and specify emotion
3. **Monitor Progress**: Check task status
4. **Download Result**: Get the generated video

### Example: cURL Commands

```bash
# 1. Health check
curl http://localhost:8000/api/health

# 2. Start synthesis task
curl -X POST "http://localhost:8000/api/v1/synthesis/start" \
  -F "image=@examples/img/female.png" \
  -F "audio=@examples/wav/female-zh.wav" \
  -F "emotion=happy" \
  -F "ref_scale=3.0" \
  -F "emo_scale=6.0"

# 3. Check status (replace {task_id} with actual ID)
curl http://localhost:8000/api/v1/synthesis/{task_id}/status

# 4. Download result when completed
curl -o result.mp4 http://localhost:8000/api/v1/synthesis/{task_id}/download
```

### Example: Python Client

```python
import requests
import time

# Start synthesis task
with open('portrait.jpg', 'rb') as img, open('speech.wav', 'rb') as aud:
    files = {'image': img, 'audio': aud}
    data = {'emotion': 'happy', 'ref_scale': 3.0, 'emo_scale': 6.0}
    
    response = requests.post(
        'http://localhost:8000/api/v1/synthesis/start',
        files=files, data=data
    )
    task = response.json()
    task_id = task['task_id']
    print(f"Task started: {task_id}")

# Monitor progress
while True:
    status_response = requests.get(
        f'http://localhost:8000/api/v1/synthesis/{task_id}/status'
    )
    status = status_response.json()
    print(f"Status: {status['status']}")
    
    if status['status'] == 'COMPLETED':
        # Download result
        result = requests.get(
            f'http://localhost:8000/api/v1/synthesis/{task_id}/download'
        )
        with open('result.mp4', 'wb') as f:
            f.write(result.content)
        print("Video downloaded successfully!")
        break
    elif status['status'] == 'FAILED':
        print(f"Task failed: {status['error']}")
        break
    
    time.sleep(5)  # Wait 5 seconds before next check
```

## 🧪 Testing with test_api.py

The project includes a comprehensive test script to validate API functionality:

### Quick Health Check
```bash
python test_api.py health
```

### Full Functionality Test
```bash
python test_api.py
```

### Test Categories

The test script validates:
- **Service Health**: Check if the API server is running
- **Task Creation**: Verify task startup functionality
- **Status Monitoring**: Test status query endpoints
- **File Download**: Validate result download functionality
- **Error Handling**: Test error scenarios and edge cases

### Test Output Example
```bash
$ python test_api.py
Testing DICE-Talk API Service...

✅ Health Check: Service is healthy
✅ Task Start: Task created successfully (ID: abc123...)
✅ Status Check: Task status retrieved
⏳ Processing: Task in progress...
✅ Download: Result video downloaded (2.3MB)

All tests passed! 🎉
```

## 📁 Project Structure

```
DICE-Talk/
├── api/                 # FastAPI service code
│   ├── server.py        # Main FastAPI server
│   ├── models.py        # Data models
│   ├── task_manager.py  # Task management
│   └── utils.py         # Utility functions
├── config/
│   └── api_config.yaml  # API configuration
├── doc/
│   └── api_docs.md      # Detailed API documentation
├── src/                 # Core DICE-Talk source code
├── checkpoints/         # Model checkpoints
├── examples/            # Sample images and audio
├── uploads/             # Upload directory
├── outputs/             # Output directory
├── logs/                # Log files
├── start_server.sh      # Startup script
├── test_api.py          # API test script
├── requirements.txt     # Core dependencies
├── requirements_api.txt # API dependencies
└── README.md            # This file
```

## 🎨 Supported Emotions

- **Neutral**: Natural, default expression
- **Happy**: Joyful, smiling expression
- **Sad**: Melancholic, downcast expression  
- **Angry**: Aggressive, frowning expression
- **Surprised**: Wide-eyed, amazed expression

## ⚙️ Configuration

### API Settings
Edit `config/api_config.yaml` to customize:
- Server host and port
- GPU device settings
- Task concurrency limits
- File size restrictions

### Performance Tuning
- Adjust `max_workers` for concurrent tasks
- Modify `inference_steps` for quality vs speed
- Configure memory management settings

## 🔗 Citations

### Original DICE-Talk Paper
```bibtex
@article{tan2025dicetalk,
  title={Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation}, 
  author={Tan, Weipeng and Lin, Chuming and Xu, Chengming and Xu, FeiFan and Hu, Xiaobin and Ji, Xiaozhong and Zhu, Junwei and Wang, Chengjie and Fu, Yanwei},
  journal={arXiv preprint arXiv:2504.18087},
  year={2025}
}
```

### Related Work
```bibtex
@article{ji2024sonic,
  title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation},
  author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
  journal={arXiv preprint arXiv:2411.16331},
  year={2024}
}

@article{ji2024realtalk,
  title={Realtalk: Real-time and realistic audio-driven face generation with 3d facial prior-guided identity alignment network},
  author={Ji, Xiaozhong and Lin, Chuming and Ding, Zhonggan and Tai, Ying and Zhu, Junwei and Hu, Xiaobin and Luo, Donghao and Ge, Yanhao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2406.18284},
  year={2024}
}
```

## 📄 License

This project follows the same license as the original DICE-Talk project: CC BY-NC-SA-4.0. For non-commercial use only.

## 🙏 Acknowledgments

This API service is built upon the excellent work of the original [DICE-Talk project](https://github.com/toto222/DICE-Talk). All core algorithms, models, and techniques are derived from their research. We extend our gratitude to the original authors for making their work available.

## 🔗 Links

- **Original DICE-Talk Project**: https://github.com/toto222/DICE-Talk
- **Project Homepage**: https://toto222.github.io/DICE-Talk/
- **Paper**: https://arxiv.org/abs/2504.18087
- **API Documentation**: See `doc/api_docs.md` for detailed API specifications

---

*This FastAPI backend service provides a modern REST API interface for the groundbreaking DICE-Talk technology, enabling easy integration of emotional talking portrait generation into web applications and services.* 