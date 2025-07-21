import pandas as pd
import os 
import numpy as np 
from scipy.optimize import linear_sum_assignment

def add_gt_to_predictions(pred_csv_path, metadata_csv_path, out_csv_path):
    """
    Merge prediction and metadata CSVs on `frame` and save output with GT columns.
    """
    # Read CSVs
    pred_df = pd.read_csv(pred_csv_path)
    meta_df = pd.read_csv(metadata_csv_path, header=None)
    meta_df.columns = ['frame_index', 'col1', 'col2', 'gt_azimuth', 'gt_elevation']

    # Ensure frame column is int
    pred_df['frame_index'] = pred_df['frame_index'].astype(int)
    meta_df['frame_index'] = meta_df['frame_index'].astype(int)

    # Rename metadata columns
    meta_df = meta_df.rename(columns={
        'azimuth': 'gt_azimuth',
        'elevation': 'gt_elevation'
    })

    # Merge on frame
    merged_df = pred_df.merge(
        meta_df[['frame_index', 'gt_azimuth', 'gt_elevation']],
        on='frame_index',
        how='left'
    )

    merged_df.to_csv(out_csv_path, index=False)
    # Remove original predictions file
    if os.path.exists(pred_csv_path):
        os.remove(pred_csv_path)

def add_gt_with_hungarian(pred_csv, gt_csv, output_csv):
    """
    Match predictions to ground truth using Hungarian matching per frame,
    and write a new CSV with GT rows + matched predictions.
    Columns: frame_index, col1, col2, gt_azimuth, gt_elevation, azimuth, elevation
    """
    df_pred = pd.read_csv(pred_csv)
    df_gt = pd.read_csv(gt_csv, header=None)
    df_gt.columns = ['frame_index', 'col1', 'col2', 'gt_azimuth', 'gt_elevation']

    df_gt['azimuth'] = np.nan
    df_gt['elevation'] = np.nan

    for frame in df_gt['frame_index'].unique():
        gt_rows = df_gt[df_gt['frame_index'] == frame]
        pred_rows = df_pred[df_pred['frame_index'] == frame]

        if gt_rows.empty or pred_rows.empty:
            continue

        gt_az = gt_rows['gt_azimuth'].values
        gt_el = gt_rows['gt_elevation'].values
        pred_az = pred_rows['azimuth'].values
        pred_el = pred_rows['elevation'].values

        # compute cost matrix
        cost = np.zeros((len(gt_az), len(pred_az)))
        for i in range(len(gt_az)):
            for j in range(len(pred_az)):
                cost[i, j] = angular_error(pred_az[j], pred_el[j], gt_az[i], gt_el[i])

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # assign matched predictions
        for i, j in zip(row_ind, col_ind):
            df_gt.loc[gt_rows.index[i], 'azimuth'] = ((pred_az[j] + 180) % 360) - 180
            df_gt.loc[gt_rows.index[i], 'elevation'] = pred_el[j]

    df_gt.to_csv(output_csv, index=False)
    if os.path.exists(pred_csv):
        os.remove(pred_csv)
    print(f"✅ Saved GT with matched predictions: {output_csv}")


def angular_error(azi_pred, ele_pred, azi_gt, ele_gt):
    """
    Compute angular error (degrees) between predicted and ground truth directions.
    Works for scalars or pandas Series.
    """
    pred_vec = sph2cart(azi_pred, ele_pred)
    gt_vec = sph2cart(azi_gt, ele_gt)

    # if input is a DataFrame/Series, ensure axis=1
    dot = np.sum(pred_vec * gt_vec, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)  # ensure valid domain for arccos
    angle = np.degrees(np.arccos(dot))
    return angle


def sph2cart(azimuth, elevation):
    az = np.atleast_1d(np.radians(azimuth))
    el = np.atleast_1d(np.radians(elevation))
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=1)

def evaluate_group(df, name='Overall'):
    mae = df['angular_error'].mean()
    std = df['angular_error'].std()
    success_rate = (df['angular_error'] <= 20).mean() * 100
    print(f"\n📊 Results for {name}:")
    print(f"  MAE: {mae:.2f}°")
    print(f"  Std Dev: {std:.2f}°")
    print(f"  Success Rate (≤20°): {success_rate:.2f}%")

def bin_and_evaluate(df, bin_edges):
    df['dist_bin'] = pd.cut(df['gt_distance'], bins=bin_edges, include_lowest=True)
    for b, group in df.groupby('dist_bin'):
        evaluate_group(group, f"Distance Bin {b}")