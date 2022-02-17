import pandas as pd
import argparse
import subprocess

plt_template = {
    "time": r"""set terminal postscript "Times, 20" eps color dashed size 6,2
set output "{filename}.ps"

set style fill   solid 1.00 border lt -1
set key horizontal top Left left
set border 3
set key autotitle columnhead reverse font "Courier,17"
set style histogram clustered gap 1 title textcolor lt -1
set datafile missing '-'
set style data histograms
set xtics border in scale 0,0 nomirror norotate autojustify
set ytics nomirror
set logscale y

set xlabel "Dataset" font "Times, 24"
set ylabel "Time/ms" font "Times, 24"
set yrange [1:90000]

to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.1f", x) : sprintf("%d", x)): (x == -1 ? "N/A": "OOM")
y_pos(x) = (x > 0) ? x * 1.1 : 1.1
y_val(x) = (x > 0) ? x : 0

set offset -0.3, -0.3, 0, 0

NO_ANIMATION = 1
plot '{filename}.dat' using 2:xtic(1) fs p 3 fc ls 5 lw 3 ti col,\
    '' using ($0-.36):(y_pos($2)):(to_str($2)) with labels notitle rotate left font ",16",\
    '' u 3 fs p 4 fc ls 3 lw 3 ti col,\
    '' using ($0-.18):(y_pos($3)):(to_str($3)) with labels notitle rotate left font ",16",\
    '' u 4 fs p 5 fc ls 4 lw 3 ti col,\
    '' using ($0+.0):(y_pos($4)):(to_str($4)) with labels notitle rotate left font ",16", \
    '' u 5 fs p 10 fc ls 2 lw 3 ti col,\
    '' using ($0+.18):(y_pos($5)):(to_str($5)) with labels notitle rotate left font ",16",\
    '' u 6 fs p 6 fc ls 6 lw 3 ti col,\
    '' using ($0+.36):(y_pos($6)):(to_str($6)) with labels notitle rotate left font ",16"
    """,
    "mem": r"""set terminal postscript "Times, 20" eps color dashed size 7,2.8
set output "{filename}.ps"

set style fill   solid 1.00 border lt -1
set key fixed left top vertical Left noreverse noenhanced autotitle nobox
set border 3
set key autotitle columnhead reverse font "Courier,24"
set style histogram clustered gap 1 title textcolor lt -1
set datafile missing '-'
set style data histograms
set xtics border in scale 0,0 nomirror norotate autojustify
set ytics nomirror

set xlabel "Dataset" font "Times, 24"
set ylabel "GPU Memory Usage/MB" font "Times, 24"
set yrange [0:16000]

set offset -0.3, -0.3, 0, 0

to_str(x) = (x > 0) ? ((x < 10) ? sprintf("%.1f", x) : sprintf("%d", x)): (x == -1 ? "N/A": "OOM")
y_pos(x) = (x > 0) ? x * 1.1 : 1.1
y_val(x) = (x > 0) ? x : 0

NO_ANIMATION = 1
plot '{filename}.dat' using 2:xtic(1) fs p 3 fc ls 5 lw 3 ti col,\
    '' using ($0-.36):($2+50):($2>0 ? sprintf("%3d",$2): "OOM") with labels notitle rotate left font ",20",\
    '' u 3 fs p 4 fc ls 3 lw 3 ti col,\
    '' using ($0-.18):($3+50):($3>0 ? sprintf("%3d",$3): "OOM") with labels notitle rotate left font ",20",\
    '' u 4 fs p 5 fc ls 4 lw 3 ti col,\
    '' using ($0):($4+50):($4>0 ? sprintf("%3d",$4): "OOM") with labels notitle rotate left font ",20", \
    '' u 5 fs p 10 fc ls 2 lw 3 ti col,\
    '' using ($0+.18):($5+50):($5>0 ? sprintf("%3d",$5): "OOM") with labels notitle rotate left font ",20",\
    '' u 6 fs p 6 fc ls 6 lw 3 ti col,\
    '' using ($0+.36):($6+50):($6>0 ? sprintf("%3d",$6): "OOM") with labels notitle rotate left font ",20",
    """
}


def gnuplot_str(metric, filename):
    return plt_template[metric].format(filename=filename)


def dat_str(metric, log):
    def to_str(arr):
        return ["0.0" if elem is None else str(elem) for elem in arr]
    ret = "\t".join(["Dataset"] + list(log.index.levels[0])) + "\n"
    for dataset in log.columns:
        ret += "\t".join([dataset] + to_str([log[dataset][tag, metric]
                         for tag in log.index.levels[0]])) + "\n"
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Graphiler performance visualization")
    parser.add_argument("--model", type=str, help="name of the model", required=True)
    parser.add_argument("--time", action="store_true", help="visualize the execution time")
    parser.add_argument("--mem", action="store_true", help="visualize the memory footprint")
    args = parser.parse_args()
    metrics = ["time", "mem"]

    if args.time == args.mem:
        raise RuntimeError("must specify time or mem")
    else:
        metric = metrics[args.mem]
        filename = "_".join([args.model, metric])

    log = pd.read_pickle("output/{}.pkl".format(args.model))

    with open("{}.plt".format(filename), "w") as f:
        f.write(gnuplot_str(metric, filename))

    with open("{}.dat".format(filename), "w") as f:
        f.write(dat_str(metric, log))

    subprocess.run(["gnuplot", "{}.plt".format(filename)])
    subprocess.run(["epstopdf", "{}.ps".format(filename)])
    subprocess.run(["mv", "{}.pdf".format(filename), "output/{}.pdf".format(filename)])
    subprocess.run(["rm", "{}.plt".format(filename),
                   "{}.ps".format(filename), "{}.dat".format(filename)])
