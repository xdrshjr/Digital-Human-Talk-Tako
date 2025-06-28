#!/usr/bin/env python3
"""
DICE-Talk API Test Script
测试API的各个功能接口
"""

import requests
import time
import os
import sys
from pathlib import Path


class DiceTalkAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def start_synthesis(self, image_path, audio_path, emotion="happy", 
                       ref_scale=3.0, emo_scale=6.0, crop=False, duration=None,
                       inference_steps=20, fps=24, seed=None):
        """启动合成任务"""
        try:
            with open(image_path, 'rb') as img, open(audio_path, 'rb') as aud:
                files = {
                    'image': (os.path.basename(image_path), img, 'image/png' if image_path.endswith('.png') else 'image/jpeg'),
                    'audio': (os.path.basename(audio_path), aud, 'audio/wav' if audio_path.endswith('.wav') else 'audio/mpeg')
                }
                data = {
                    'emotion': emotion,
                    'ref_scale': ref_scale,
                    'emo_scale': emo_scale,
                    'crop': crop,
                    'inference_steps': inference_steps,
                    'fps': fps
                }
                
                # Add optional parameters
                if duration is not None:
                    data['duration'] = duration
                if seed is not None:
                    data['seed'] = seed
                
                response = requests.post(
                    f"{self.base_url}/api/v1/synthesis/start", 
                    files=files, 
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_task_status(self, task_id):
        """查询任务状态"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/synthesis/{task_id}/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def download_result(self, task_id, output_path):
        """下载结果视频"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/synthesis/{task_id}/download",
                timeout=60
            )
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return {"success": True, "path": output_path}
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_completion(self, task_id, max_wait_time=600):
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_task_status(task_id)
            
            if "error" in status:
                return status
            
            print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.2%}")
            
            if status["status"] == "COMPLETED":
                return status
            elif status["status"] == "FAILED":
                return {"error": f"Task failed: {status.get('message', 'Unknown error')}"}
            
            time.sleep(5)
        
        return {"error": "Task timeout"}


def test_api():
    """测试API功能"""
    client = DiceTalkAPIClient()
    
    print("🔍 DICE-Talk API 功能测试")
    print("=" * 50)
    
    # 1. 健康检查
    print("\n1. 健康检查测试...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ 健康检查失败: {health['error']}")
        print("请确保API服务已启动: ./start_server.sh")
        return False
    else:
        print(f"✅ 服务状态: {health['status']}")
        print(f"   GPU可用: {health['gpu_available']}")
        print(f"   模型已加载: {health['model_loaded']}")
        print(f"   版本: {health['version']}")
    
    # 检查示例文件
    example_image = "examples/img/nazha.png"
    example_audio = "examples/wav/female-zh.wav"
    
    if not os.path.exists(example_image):
        print(f"\n❌ 示例图片文件不存在: {example_image}")
        print("请确保examples目录包含示例文件")
        return False
    
    if not os.path.exists(example_audio):
        print(f"\n❌ 示例音频文件不存在: {example_audio}")
        print("请确保examples目录包含示例文件")
        return False
    
    # 2. 启动合成任务
    print("\n2. 启动合成任务测试...")
    task_result = client.start_synthesis(
        image_path=example_image,
        audio_path=example_audio,
        emotion="happy",
        ref_scale=3.0,
        emo_scale=6.0
    )
    
    if "error" in task_result:
        print(f"❌ 启动任务失败: {task_result['error']}")
        return False
    else:
        print(f"✅ 任务创建成功")
        print(f"   任务ID: {task_result['task_id']}")
        print(f"   状态: {task_result['status']}")
        task_id = task_result['task_id']
    
    # 3. 查询任务状态
    print("\n3. 任务状态查询测试...")
    print("等待任务完成（这可能需要几分钟）...")
    
    status_result = client.wait_for_completion(task_id)
    
    if "error" in status_result:
        print(f"❌ 任务处理失败: {status_result['error']}")
        return False
    else:
        print(f"✅ 任务完成")
        print(f"   最终状态: {status_result['status']}")
        print(f"   进度: {status_result.get('progress', 1):.2%}")
    
    # 4. 下载结果
    print("\n4. 结果下载测试...")
    output_file = f"test_result_{task_id[:8]}.mp4"
    
    download_result = client.download_result(task_id, output_file)
    
    if "error" in download_result:
        print(f"❌ 下载失败: {download_result['error']}")
        return False
    else:
        print(f"✅ 下载成功")
        print(f"   文件路径: {download_result['path']}")
        print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    print("\n🎉 所有测试通过！API功能正常")
    return True


def quick_health_check():
    """快速健康检查"""
    client = DiceTalkAPIClient()
    health = client.health_check()
    
    if "error" in health:
        print(f"❌ 服务不可用: {health['error']}")
        return False
    else:
        print(f"✅ 服务正常运行")
        print(f"   状态: {health['status']}")
        print(f"   GPU: {'可用' if health['gpu_available'] else '不可用'}")
        print(f"   模型: {'已加载' if health['model_loaded'] else '未加载'}")
        return True


def test_duration_functionality():
    """测试duration参数功能"""
    client = DiceTalkAPIClient()
    
    print("🎬 测试duration参数功能")
    print("=" * 50)
    
    # 健康检查
    health = client.health_check()
    if "error" in health:
        print(f"❌ 服务不可用: {health['error']}")
        return False
    
    example_image = "examples/img/nazha.png"
    example_audio = "examples/wav/female-zh.wav"
    
    if not os.path.exists(example_image) or not os.path.exists(example_audio):
        print("❌ 示例文件不存在")
        return False
    
    # 测试duration=10秒
    print("\n测试duration=10秒的视频生成...")
    task_result = client.start_synthesis(
        image_path=example_image,
        audio_path=example_audio,
        emotion="happy",
        duration=2.0,
        ref_scale=2.0,
        emo_scale=5.0
    )
    
    if "error" in task_result:
        print(f"❌ 任务创建失败: {task_result['error']}")
        return False
    
    print(f"✅ 任务创建成功 (ID: {task_result['task_id']})")
    
    # 等待完成
    status_result = client.wait_for_completion(task_result['task_id'])
    if "error" in status_result:
        print(f"❌ 任务失败: {status_result['error']}")
        return False
    
    # 下载结果
    output_file = f"test_duration_{task_result['task_id'][:8]}.mp4"
    download_result = client.download_result(task_result['task_id'], output_file)
    
    if "error" in download_result:
        print(f"❌ 下载失败: {download_result['error']}")
        return False
    
    print(f"✅ Duration测试完成，文件: {output_file}")
    print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            quick_health_check()
        elif sys.argv[1] == "duration":
            test_duration_functionality()
        elif sys.argv[1] == "help":
            print("使用方法:")
            print("  python test_api.py          # 运行完整的API测试")
            print("  python test_api.py health   # 快速健康检查")
            print("  python test_api.py duration # 测试duration参数功能")
            print("  python test_api.py help     # 显示帮助信息")
        else:
            print("未知参数，使用 'help' 查看使用方法")
    else:
        test_api() 