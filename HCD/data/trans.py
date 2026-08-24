# Read original score matrix
with open("data_o.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Write: only keep 0 / 1, skip nan
with open("data.csv", "w", encoding="utf-8") as f:
    f.write("user_id,exer_id,score\n")
    for user_id, line in enumerate(lines):
        scores = line.split(" ")

        for exer_id, score in enumerate(scores):
            if score == "0.0000":
                f.write(f"{user_id},{exer_id},0\n")
            elif score == "1.0000":
                f.write(f"{user_id},{exer_id},1\n")

# Statistics
num_users = len(lines)
num_exer = len(lines[0].split(" ")) if num_users > 0 else 0

print("Conversion complete! Skipped nan, kept only 0/1")
print(f"Number of students: {num_users}")
print(f"Number of exercises: {num_exer}")