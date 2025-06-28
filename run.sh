export https_proxy=http://127.0.0.1:7897;
export http_proxy=http://127.0.0.1:7897;
export all_proxy=socks5://127.0.0.1:7897;

export CUDA_VISIBLE_DEVICES=0

python3 demo.py --image_path './relevant-materials/base-person.png' --audio_path './relevant-materials/tts_b70c0bb3-96bc-4dec-aef6-aefe136b17e4.mp3'\
  --emotion_path './relevant-materials/happy.npy' --output_path './output/test.mp4'