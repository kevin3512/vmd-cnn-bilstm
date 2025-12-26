from main import load_data, vmd_decompose, plot_true_vs_vmd_sum


if __name__ == '__main__':
    series, scaler = load_data("安徽数据集.xlsx")
    imfs = vmd_decompose(series)
    # 保存到文件
    plot_true_vs_vmd_sum(series, imfs, test_percent=0.2, save_path="test_true_vs_vmd_sum.png")
    print('Saved plot: test_true_vs_vmd_sum.png')
