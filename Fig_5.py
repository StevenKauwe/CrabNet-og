from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.composition import _element_composition
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=False)
data_type_torch = torch.float32


# %%
all_symbols = [
    "None",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

color = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

mat_prop = "oxides"
model = Model(CrabNet().to(compute_device), model_name=f"{mat_prop}")
model.classification = True
model.load_network(f"{mat_prop}.pth")
model.model_name = f"{mat_prop}"

test_data = rf"data/benchmark_data/{mat_prop}/train.csv"
model.load_data(test_data, batch_size=2**0)

model.model.eval()
model.model.avg = False

quaternaries_dict = defaultdict(list)
for data in model.data_loader.dataset:
    keys = _element_composition(data[2]).keys()
    if len(keys) == 4 and "O" in keys:
        quaternaries_dict[tuple(sorted(keys))].append(data)


def analyze_quaternary(key, systems, res=150):
    if "P" not in key or "Fe" not in key or "O" not in key:
        return

    print(f"Analyzing: {key}")

    with torch.no_grad():
        X, y, formula = systems[0]  # Use the first system as a template
        slist = X[: len(X) // 2].tolist()
        slist = [int(x[0]) for x in slist]
        flist = X[len(X) // 2 :].tolist()
        flist = [x[0] for x in flist]

        # Identify indices for P, Fe, O, and Other
        p_idx = slist.index(all_symbols.index("P"))
        fe_idx = slist.index(all_symbols.index("Fe"))
        o_idx = slist.index(all_symbols.index("O"))
        other_idx = [i for i in range(len(slist)) if i not in [p_idx, fe_idx, o_idx]][0]

        # Determine fixed O and X (Other) content
        fixed_o = flist[o_idx]
        fixed_x = flist[other_idx]
        other_element = all_symbols[slist[other_idx]]

        # Find the bounds for P and Fe fractions from actual systems
        p_fracs = [system[0][len(system[0]) // 2 + p_idx][0] for system in systems]
        fe_fracs = [system[0][len(system[0]) // 2 + fe_idx][0] for system in systems]
        p_min, p_max = min(p_fracs), max(p_fracs)
        fe_min, fe_max = min(fe_fracs), max(fe_fracs)

        frac_tracker = []
        pred_tracker = []
        uncertainty_tracker = []
        elem_trackers = [[], [], [], []]  # P, Fe, Other, O

        for i in range(res):
            for j in range(res):
                p_frac = p_min + (p_max - p_min) * i / (res - 1)
                fe_frac = fe_min + (fe_max - fe_min) * j / (res - 1)
                other_frac = fixed_x
                o_frac = fixed_o
                if p_frac + fe_frac + other_frac > 1:
                    continue

                frac = flist.copy()
                frac[p_idx] = p_frac
                frac[fe_idx] = fe_frac
                frac[other_idx] = other_frac
                frac[o_idx] = o_frac

                X = torch.tensor(slist + frac).unsqueeze(0)
                src, frac = X.chunk(2, dim=1)

                src = src.to(compute_device, dtype=torch.long)
                frac = frac.to(compute_device, dtype=data_type_torch)

                output = model.model.forward(src, frac)
                mask = (src == 0).unsqueeze(-1).repeat(1, 1, 3)
                output = output.masked_fill(mask, 0)
                prob = output[:, :, -1:]
                output = output[:, :, :2]
                probability = torch.ones_like(output)
                probability[:, :, :1] = torch.sigmoid(prob)
                output = output * probability

                prediction, uncertainty = output.chunk(2, dim=-1)
                prediction = model.scaler.unscale(prediction)
                prediction = prediction * ~mask
                if model.classification:
                    prediction = torch.sigmoid(prediction)
                uncertainty = uncertainty * ~mask
                uncertainty_mean = (uncertainty * ~mask).sum() / (~mask).sum()
                uncertainty = torch.exp(uncertainty_mean) * model.scaler.std

                frac_tracker.append([p_frac, fe_frac])
                pred_tracker.append(prediction.mean().cpu().item())
                uncertainty_tracker.append(uncertainty.cpu().item())

                # Extract element-wise contributions
                for idx, tracker in zip(
                    [p_idx, fe_idx, other_idx, o_idx], elem_trackers
                ):
                    tracker.append(prediction[0, idx, 0].cpu().item())

        frac_tracker = np.array(frac_tracker)
        pred_tracker = np.array(pred_tracker)
        uncertainty_tracker = np.array(uncertainty_tracker)
        elem_trackers = [np.array(tracker) for tracker in elem_trackers]

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(
            frac_tracker[:, 0],
            frac_tracker[:, 1],
            c=pred_tracker,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(sc, label="Prediction")
        plt.xlabel("P fraction")
        plt.ylabel("Fe fraction")
        plt.xlim(p_min, p_max)
        plt.ylim(fe_min, fe_max)

        # Add true labels for all systems
        for system in systems:
            X, y, formula = system
            p_frac = X[len(X) // 2 + p_idx][0]
            fe_frac = X[len(X) // 2 + fe_idx][0]
            if y.item() == 0:
                plt.plot(p_frac, fe_frac, "bs", markersize=10, mew=2, mfc="none")
            else:
                plt.plot(p_frac, fe_frac, "yo", markersize=10, mew=2, mfc="none")

        plt.title(
            f"Prediction for {key} system\nFixed {other_element}: {fixed_x:.2f}, O: {fixed_o:.2f}"
        )
        plt.savefig(
            f"figures/Quaternary_prediction_{'-'.join(key)}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # plt.figure(figsize=(10, 8))
        # for i, (elem, tracker) in enumerate(
        #     zip(["P", "Fe", other_element, "O"], elem_trackers)
        # ):
        #     plt.plot(
        #         range(len(tracker)),
        #         tracker,
        #         label=f"{elem} contribution",
        #         color=color[i],
        #         linewidth=2,
        #     )
        # plt.fill_between(
        #     range(len(pred_tracker)),
        #     pred_tracker - uncertainty_tracker,
        #     pred_tracker + uncertainty_tracker,
        #     alpha=0.3,
        #     color="gray",
        # )
        # plt.plot(
        #     range(len(pred_tracker)),
        #     pred_tracker,
        #     label="Total Prediction",
        #     color="black",
        #     linewidth=2,
        # )
        # plt.xlabel("Composition Index")
        # plt.ylabel("Contribution / Prediction")
        # plt.title(
        #     f"Element Contributions for {key} system\nFixed {other_element}: {fixed_x:.2f}, O: {fixed_o:.2f}"
        # )
        # plt.legend()
        # plt.savefig(
        #     f"figures/Quaternary_contributions_{'-'.join(key)}.png",
        #     bbox_inches="tight",
        #     dpi=300,
        # )
        # plt.close()


for key, systems in quaternaries_dict.items():
    analyze_quaternary(key, systems)
