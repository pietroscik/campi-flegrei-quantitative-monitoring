import subprocess

def test_pipeline_runs():
    result = subprocess.run(["python", "run_pipeline.py"])
    assert result.returncode == 0
