import splitfolders

# Input path-ah double training_set irukara mari update panikonga
input_folder = r"C:\Users\Karthikeyan\Downloads\ML_Project\archive (1)\training_set\training_set"
output_folder = r"C:\Users\Karthikeyan\Downloads\ML_Project\split folder"

splitfolders.ratio(input_folder, output=output_folder,
                   seed=42, ratio=(0.7, 0.2, 0.1), move=False)

print("✅ Dataset split completed!")