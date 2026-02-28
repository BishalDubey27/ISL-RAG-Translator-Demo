#!/usr/bin/env python3
"""
Complete project status checker for ISL RAG Translator Demo
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version compatible")
        return True

def check_virtual_environment():
    """Check if virtual environment is active"""
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if venv_active:
        print("🔧 Virtual Environment: ✅ Active")
        print(f"   Path: {sys.prefix}")
        return True
    else:
        print("🔧 Virtual Environment: ⚠️  Not active")
        print("   Recommendation: Activate with 'venv\\Scripts\\activate'")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'sentence_transformers', 'gtts', 'numpy', 
        'requests', 'watchdog', 'tqdm', 'pillow'
    ]
    
    print("📦 Dependencies:")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"   Missing: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if all required files and directories exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'simple_audio_pipeline.py',
        'check_video_status.py',
        'templates/enhanced_index.html',
        'knowledge_base/metadata.json',
        'knowledge_base/videos/',
    ]
    
    print("📁 Project Structure:")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                count = len([f for f in os.listdir(file_path) if f.endswith('.mp4')])
                print(f"   ✅ {file_path} ({count} videos)")
            else:
                print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_data_status():
    """Check status of videos, audio, and metadata"""
    print("📊 Data Status:")
    
    # Videos
    video_dir = "knowledge_base/videos"
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"   🎬 Videos: {len(videos)} files")
    else:
        print("   🎬 Videos: ❌ Directory not found")
        return False
    
    # Audio
    audio_dir = "knowledge_base/generated_audio"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        print(f"   🔊 Audio: {len(audio_files)} files")
    else:
        print("   🔊 Audio: ❌ Directory not found")
    
    # Metadata
    metadata_file = "knowledge_base/metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"   📋 Metadata: {len(metadata)} entries")
    else:
        print("   📋 Metadata: ❌ File not found")
        return False
    
    return True

def check_flask_app():
    """Check if Flask app can be imported and configured"""
    print("🌐 Flask Application:")
    
    try:
        # Try to import the app
        sys.path.insert(0, '.')
        from app import app, load_components
        print("   ✅ App import successful")
        
        # Try to load components
        load_components()
        print("   ✅ Components loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print("💻 System Resources:")
    
    try:
        import psutil
        
        # Memory
        memory = psutil.virtual_memory()
        print(f"   🧠 Memory: {memory.percent}% used ({memory.available // (1024**3)} GB available)")
        
        # Disk space
        disk = psutil.disk_usage('.')
        print(f"   💾 Disk: {disk.percent}% used ({disk.free // (1024**3)} GB available)")
        
        # CPU
        cpu = psutil.cpu_percent(interval=1)
        print(f"   ⚡ CPU: {cpu}% usage")
        
        return True
        
    except ImportError:
        print("   ⚠️  psutil not available for resource monitoring")
        return True

def generate_recommendations():
    """Generate recommendations based on checks"""
    print("\n💡 Recommendations:")
    
    recommendations = []
    
    # Check if virtual environment is active
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        recommendations.append("Activate virtual environment: venv\\Scripts\\activate")
    
    # Check if audio files exist
    if not os.path.exists("knowledge_base/generated_audio"):
        recommendations.append("Generate audio files: python simple_audio_pipeline.py")
    
    # Check if Flask app is ready
    try:
        import flask
        recommendations.append("Start the application: python app.py")
        recommendations.append("Visit: http://localhost:5000/enhanced")
    except ImportError:
        recommendations.append("Install dependencies: pip install -r requirements.txt")
    
    if not recommendations:
        recommendations.append("✅ System is ready! Run 'python app.py' to start")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

def main():
    """Main status check function"""
    print("🎬 ISL RAG Translator Demo - Project Status Check")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Working Directory: {os.getcwd()}")
    print()
    
    checks = [
        check_python_version(),
        check_virtual_environment(),
        check_dependencies(),
        check_project_structure(),
        check_data_status(),
        check_flask_app(),
        check_system_resources()
    ]
    
    passed_checks = sum(checks)
    total_checks = len(checks)
    
    print(f"\n📊 Status Summary: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! System is ready.")
    elif passed_checks >= total_checks - 1:
        print("⚠️  Minor issues detected. System mostly ready.")
    else:
        print("❌ Multiple issues detected. Please address before running.")
    
    generate_recommendations()
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)