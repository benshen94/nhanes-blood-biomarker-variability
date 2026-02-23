import pandas as pd
import os

COLUMNS_TO_REMOVE = [
    "z_n","z_min","z_q1","z_q25","z_median","z_mad","z_se","z_q75","z_q3","z_max","z_mean","z_sd","z_cv",
    "z_log_mean","z_log_sd","z_log_cv","z_log_se","z_log_n","z_log_q1","z_log_q25","z_log_median",
    "z_log_q75","z_log_q3","z_q1_error","z_q25_error","z_median_error","z_q75_error","z_q3_error",
    "z_log_q1_error","z_log_q25_error","z_log_median_error","z_log_q75_error","z_log_q3_error",
    "ref_mean_n","ref_mean_min","ref_mean_q1","ref_mean_q25","ref_mean_median","ref_mean_mad","ref_mean_se",
    "ref_mean_q75","ref_mean_q3","ref_mean_max","ref_mean_mean","ref_mean_sd","ref_mean_cv",
    "ref_mean_log_mean","ref_mean_log_sd","ref_mean_log_cv","ref_mean_log_se","ref_mean_log_n",
    "ref_mean_log_q1","ref_mean_log_q25","ref_mean_log_median","ref_mean_log_q75","ref_mean_log_q3",
    "ref_mean_q1_error","ref_mean_q25_error","ref_mean_median_error","ref_mean_q75_error","ref_mean_q3_error",
    "ref_mean_log_q1_error","ref_mean_log_q25_error","ref_mean_log_median_error","ref_mean_log_q75_error","ref_mean_log_q3_error",
    "ref_sd_n","ref_sd_min","ref_sd_q1","ref_sd_q25","ref_sd_median","ref_sd_mad","ref_sd_se",
    "ref_sd_q75","ref_sd_q3","ref_sd_max","ref_sd_mean","ref_sd_sd","ref_sd_cv",
    "ref_sd_log_mean","ref_sd_log_sd","ref_sd_log_cv","ref_sd_log_se","ref_sd_log_n",
    "ref_sd_log_q1","ref_sd_log_q25","ref_sd_log_median","ref_sd_log_q75","ref_sd_log_q3",
    "ref_sd_q1_error","ref_sd_q25_error","ref_sd_median_error","ref_sd_q75_error","ref_sd_q3_error",
    "ref_sd_log_q1_error","ref_sd_log_q25_error","ref_sd_log_median_error","ref_sd_log_q75_error","ref_sd_log_q3_error",
    "ref_n_n","ref_n_min","ref_n_q1","ref_n_q25","ref_n_median","ref_n_mad","ref_n_se",
    "ref_n_q75","ref_n_q3","ref_n_max","ref_n_mean","ref_n_sd","ref_n_cv",
    "ref_n_log_mean","ref_n_log_sd","ref_n_log_cv","ref_n_log_se","ref_n_log_n",
    "ref_n_log_q1","ref_n_log_q25","ref_n_log_median","ref_n_log_q75","ref_n_log_q3",
    "ref_n_q1_error","ref_n_q25_error","ref_n_median_error","ref_n_q75_error","ref_n_q3_error",
    "ref_n_log_q1_error","ref_n_log_q25_error","ref_n_log_median_error","ref_n_log_q75_error","ref_n_log_q3_error",
    "log_min","log_max","log_log_mean","log_log_sd","log_log_cv","log_log_se","log_log_n",
    "log_log_q1","log_log_q25","log_log_median","log_log_q75","log_log_q3",
    "log_log_q1_error","log_log_q25_error","log_log_median_error","log_log_q75_error","log_log_q3_error",
    "age_actual_n","age_actual_min","age_actual_q1","age_actual_q25","age_actual_median","age_actual_mad","age_actual_se",
    "age_actual_q75","age_actual_q3","age_actual_max","age_actual_mean","age_actual_sd","age_actual_cv",
    "age_actual_log_mean","age_actual_log_sd","age_actual_log_cv","age_actual_log_se","age_actual_log_n",
    "age_actual_log_q1","age_actual_log_q25","age_actual_log_median","age_actual_log_q75","age_actual_log_q3",
    "age_actual_q1_error","age_actual_q25_error","age_actual_median_error","age_actual_q75_error","age_actual_q3_error",
    "age_actual_log_q1_error","age_actual_log_q25_error","age_actual_log_median_error","age_actual_log_q75_error","age_actual_log_q3_error",
]

FILES = [
    "females_all_statistics.csv",
    "males_all_statistics.csv",
]

script_dir = os.path.dirname(os.path.abspath(__file__))

for filename in FILES:
    path = os.path.join(script_dir, filename)
    df = pd.read_csv(path)
    cols_found = [c for c in COLUMNS_TO_REMOVE if c in df.columns]
    cols_missing = [c for c in COLUMNS_TO_REMOVE if c not in df.columns]
    df.drop(columns=cols_found, inplace=True)
    df.to_csv(path, index=False)
    print(f"{filename}: removed {len(cols_found)} columns, saved.")
    if cols_missing:
        print(f"  (not found / already absent: {cols_missing})")

print("Done.")
