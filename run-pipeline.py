import subprocess
import sys
import time

# --- CONFIG ---
# List of scripts to execute in exact order
PIPELINE = [
    {"file": "age_detection/face_janitor.py", "name": "🧹 Face Janitor (Data Prep)"},
    {"file": "gender_detection/auto-trainer-gender.py", "name": "🧬 Gender Brain Auto-Trainer"},
    {"file": "age_detection/auto-trainer-age-v2.py", "name": "⏳ Age Brain Auto-Trainer"}
]

def run_script(script_info):
    script_file = script_info["file"]
    script_name = script_info["name"]
    
    print("\n" + "="*50)
    print(f"🚀 STARTING: {script_name}")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    try:
        # sys.executable ensures it uses your exact current Python environment
        result = subprocess.run([sys.executable, script_file], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ SUCCESS: '{script_file}' completed in {elapsed_time:.1f} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FATAL ERROR: '{script_file}' crashed with exit code {e.returncode}.")
        print("🛑 Halting the entire pipeline to prevent data corruption.")
        return False
    except FileNotFoundError:
        print(f"\n❌ FILE NOT FOUND: Cannot find '{script_file}' in the current directory.")
        return False

def main():
    print("🤖 SDE - MASTER PIPELINE INITIALIZED 🤖")
    total_start_time = time.time()
    
    for step in PIPELINE:
        success = run_script(step)
        if not success:
            sys.exit(1)
            
        time.sleep(2)
        
    total_time = time.time() - total_start_time
    print("\n" + "★"*50)
    print(f"🎉 MASTER PIPELINE COMPLETE! Total time: {total_time / 60:.1f} minutes.")
    print("★"*50)

if __name__ == "__main__":
    main()