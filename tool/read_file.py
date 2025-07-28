import os

def save_filenames_to_txt(folder_path, output_txt_path, extensions=('.png', '.jpg', '.jpeg')):
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    # print(filenames)
    filenames.sort()  # Optional: sort alphabetically

    with open(output_txt_path, 'w') as f:
        for name in filenames:
            f.write(name + '\n')

    print(f"Saved {len(filenames)} filenames to: {output_txt_path}")

# Example usage
# t2_folder = "/media/ubuntu/Elements SE/data2025.6.3/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/img/Train/t2"
# output_txt = "/media/ubuntu/Elements SE/data2025.6.3/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/img/datatr.txt"
# t2_folder = "/media/ubuntu/Elements SE/data2025.6.3/Task10_Colon/img/Test/t2"
# output_txt = "/media/ubuntu/Elements SE/data2025.6.3/Task10_Colon/img/datate.txt"
save_filenames_to_txt(t2_folder, output_txt)