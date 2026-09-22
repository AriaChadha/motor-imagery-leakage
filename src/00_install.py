import subprocess, sys
 
PACKAGES = [
    "mne==1.7.0",
    "scikit-learn==1.5.0",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scipy==1.13.0",
    "matplotlib==3.9.0",
    "tensorflow==2.16.1",
    "simpy==4.1.1",
    "PyQt5==5.15.10",
    "markdown==3.6",
    "weasyprint==62.3",
    "tqdm==4.66.4",
]
 
if __name__ == "__main__":
    print("=" * 55)
    print("EEG Wheelchair Pipeline — Dependency Installer")
    print("=" * 55)
    for pkg in PACKAGES:
        print(f"  Installing {pkg} ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            capture_output=True, text=True
        )
        print(f"  {'[OK]' if r.returncode == 0 else '[ERROR]'} {pkg}")
        if r.returncode != 0:
            print(f"    {r.stderr.strip()}")
    print("\nDone. Run src/01_data_loader.py next.")
