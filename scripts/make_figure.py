from pathlib import Path

import matplotlib.pyplot as plt

POSE_ORDER = [0, 2, 1, 3]
REPEAT = 3

out = Path(__file__).resolve().parent.parent / "assets"

y = POSE_ORDER * REPEAT
x = range(len(y))
y_vals = range(min(POSE_ORDER), max(POSE_ORDER) + 1)
xs, ys = zip(*((x_, y_) for x_ in x for y_ in y_vals if y_ != y[x_]))

fig, ax = plt.subplots()
ax.scatter(x, y, c="r")
ax.plot(x, y, c="r")
ax.scatter(xs, ys, c="black", marker=".")
ax.vlines(
    [i * len(POSE_ORDER) - 0.5 for i in range(1, REPEAT)],
    ymin=y_vals[0],
    ymax=y_vals[-1],
    linestyles="dashed",
    color="r",
)
ax.set_xlabel("time")
ax.set_ylabel("probe pose")
ax.set_yticks(y_vals)
ax.set_aspect("equal", adjustable="box")
fig.savefig(out / "pose-time.svg", bbox_inches="tight", pad_inches=0, transparent=True)
