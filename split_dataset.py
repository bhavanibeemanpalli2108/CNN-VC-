from splitfolders import ratio

ratio(
    input="Dataset",
    output="data",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
    # 70% → Training Data
    # 15% → Validation Data
    # 15% → Test Data
)