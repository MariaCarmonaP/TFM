
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
from matplotlib.patches import ConnectionPatch

reduce_colors = {
    "wht": "wh",
    "wh-rd": "rd",
    "whrd": "rd",

    "dbu": "bu",
    "dgr": "gr",
    "whdgr": "gr",
    "whgr": "gr",
    "grdgr": "gr",
    "brbl": "br",
    "blbr": "bl",
    "dgrbr": "gr",
    "grbr": "gr",
    "drd": "rd",
    "blrd": "bl",
    "rdbl": "rd",
    "dbubl": "bu",
    "bubl": "bu",
    "dbubr": "bu",
    "bubr": "bu",
    "bk": "bl",
    "rdbr": "rd",
    "grbl": "gr",
    "whbl": "bl",
    "whbr": "br",

}

color_html_dict = {
    "ye": "LemonChiffon",
    "bu": "LightBlue",
    "rd": "indianred",
    "gr": "lightgrey",
    "bl": "dimgrey",
    "br": "peru",
    "wh": "cornsilk",
}
color_dict = {
    "ye": "Amarillo",
    "bu": "Azul",
    "rd": "Rojo",
    "gr": "Gris",
    "bl": "Negro",
    "br": "Marrón",
    "wh": "Blanco",
}
def parse_info_files(path: Path) -> tuple[dict[str, int], int, list[int]]:
    colors: dict[str, int] = {}
    fronts = 0
    dists = [0]*4
    last_colors: dict[str, int] = {}
    last_fronts = 0
    last_dists = [0]*4
    for f in sorted(path.iterdir()):
        if f.suffix != ".txt":
            continue
        file_info = parse_info_file(f)
        if file_info:
            file_colors, file_fronts, file_dists = file_info
            last_colors = file_colors
            last_fronts = file_fronts
            last_dists = file_dists
        else:
            file_colors = last_colors
            file_fronts = last_fronts
            file_dists = last_dists

        for color, amount in file_colors.items():
            if color in colors:
                colors[color] += amount
            else:
                colors[color] = amount

        fronts += file_fronts
        aux = [dist + file_dist for dist, file_dist in zip(dists, file_dists)]
        dists = aux

    return colors, fronts, dists


def parse_info_file(path: Path
                    ) -> tuple[dict[str, int], int, list[int]] | None:
    all_colors: dict[str, int] = {}
    all_fronts = 0
    all_dists = [0]*4
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
        colors = lines[0].split()
        fronts = [int(elem) for elem in lines[1].split()]
        dists = [int(elem) for elem in lines[2].split()]

    if "no" in colors:
        return None
    for i, dist in enumerate(dists):
        if dist == -1:
            continue
        if fronts[i] == 1:
            all_fronts += 1
        if colors[i] in reduce_colors:
            colors[i] = reduce_colors[colors[i]]
        if colors[i] in all_colors:
            all_colors[colors[i]] += 1
        else:
            all_colors[colors[i]] = 1
        all_dists[dist] += 1

    return all_colors, all_fronts, all_dists


def show_graph(y: ArrayLike, labels: list[str], colors: list[str], title: str):
    plt.pie(y, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.suptitle(title)
    plt.show(block=True)

def draw_bar_of_pie(n_white: int, n_others: int, other_colors: list[str], other_amounts: list[int]):
    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
   
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    overall_ratios = [n_white, n_others]
    labels = ['Blanco', 'Otros colores']
    explode = [0, 0.1]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * overall_ratios[0]
    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                         labels=labels, explode=explode, colors=["LightSteelBlue", "MediumSeaGreen"])

    bottom = 1
    width = .2

    # Adding from the top matches the legend.
    for j, (height, label) in enumerate(reversed([*zip(other_amounts, other_colors)])):
        bottom -= height
        text = color_dict[label]
        bc = ax2.bar(0, height, width, bottom=bottom, color=color_html_dict[label], label=text)

                    #alpha=0.1 + 0.25 * j)
        ax2.bar_label(bc, labels=[f"{height/n_others:.0%}"], label_type='center')

    ax2.legend()
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    bar_height = sum(other_amounts)

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, -bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(2)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(2)
    
    plt.suptitle("Colores en dataset reducido")
    plt.show()

def create_graphs(colors: dict[str, int],
                  fronts: int,
                  dist_amounts: list[int],
                  ) -> None:
    total = 0
    n_white = 0
    color_labels = []
    color_amounts = []
    for color, amount in colors.items():
        total += amount
        if color == "wh":
            n_white = amount
            continue
        color_labels.append(color)
        color_amounts.append(amount)

    not_white = total - n_white
    front_labels = ["De frente", "De espaldas"]
    front_amounts = [fronts, total - fronts]
    dist_labels = ["Cerca", "Media", "Lejos", "Muy lejos"]

    draw_bar_of_pie(n_white, not_white, color_labels, color_amounts)
    show_graph(np.array(front_amounts), labels=front_labels, colors=["lightblue", "peru"], title="Sentidos en dataset reducido")
    show_graph(np.array(dist_amounts), labels=dist_labels, colors=["#c94c4c", "#eea29a", "#b1cbbb", "#deeaee"], title="Distancias en dataset reducido")


if __name__ == "__main__":
    p = Path(r"C:\Users\sierr\Documents\Uni\TFM\data\datasets\filtered_DATASET_v2\dataset_info")
    create_graphs(*parse_info_files(p))