input_file = "requirements.txt"
output_file = "requirements_cleaned.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        # Skip lines with local paths
        if "@ file://" in line:
            continue
        f_out.write(line)


