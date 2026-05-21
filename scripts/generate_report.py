import pandas as pd

def generate():

    df = pd.read_csv("data/processed/early_warning_system.csv")

    summary = df["state"].value_counts()

    with open("reports/weekly_report.md", "w") as f:
        f.write("# Weekly Seismic Report\n\n")
        f.write(summary.to_string())

if __name__ == "__main__":
    generate()
