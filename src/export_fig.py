import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import ipdb
from utils import export_graph


if __name__ == '__main__':

    ## -_ コマンドライン引数
    parser      = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv')
    parser.add_argument('-c2', '--csv2', default=None)
    args        = parser.parse_args()
    csv_path    = args.csv
    csv2_path   = args.csv2
    ## _-

    #ipdb.set_trace()
    ## -_ 出力先
    out_dir     = os.path.join('result', 'png')
    os.makedirs(out_dir, exist_ok=True)
    csv_file    = os.path.basename(csv_path)
    csv_name    = os.path.splitext(csv_file)[0]
    title       = re.match(r'.*(epochs=\d+_batchSize=\d+).*', csv_name).group(1)
    png_file    = f'{title}.png'
    png_out     = os.path.join(out_dir, png_file)
    ## _-

    ## -_ csvを読み込み
    plt.figure(num=1, figsize=(13,4), dpi=200, tight_layout=True)
    plt.suptitle(title)
    for _csv_path in [csv_path, csv2_path]:
        if _csv_path is None:
            break
        df      = pd.read_csv(_csv_path)
        label_postfix   = '_' + re.match(r'.*(N=\d+).*', _csv_path).group(1)

        if '_N=15' == label_postfix:
            linestyle = '-'
        elif '_N=19' == label_postfix:
            linestyle = '--'
        else:
            linestyle = '-'

        export_graph(df, label_postfix, linestyle)
        ## _-

    ## -_ PNGに保存
    plt.savefig(png_out)
    print(f'[SAVE]{png_out}')
    ## _-

