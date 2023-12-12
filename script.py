from typing import List
import math
import pandas as pd
import numpy as np
from itertools import accumulate
from sklearn.metrics import mean_squared_error, log_loss


def cross_comparison(revlogs, algoA, algoB):
    if algoA != algoB:
        cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()
        bin_algo = (
            algoA,
            algoB,
        )
        pair_algo = [(algoA, algoB), (algoB, algoA)]
    else:
        cross_comparison_record = revlogs[[f"R ({algoA})", "y"]].copy()
        bin_algo = (algoA,)
        pair_algo = [(algoA, algoA)]

    def get_bin(x, bins=20):
        return (
            np.log(np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1)
            / np.log(bins)
        ).round(3)

    for algo in bin_algo:
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    universal_metric_list = []

    for algoA, algoB in pair_algo:
        cross_comparison_group = cross_comparison_record.groupby(by=f"{algoA}_bin").agg(
            {"y": ["mean"], f"{algoB}_B-W": ["mean"], f"R ({algoB})": ["mean", "count"]}
        )
        universal_metric = mean_squared_error(
            y_true=cross_comparison_group["y", "mean"],
            y_pred=cross_comparison_group[f"R ({algoB})", "mean"],
            sample_weight=cross_comparison_group[f"R ({algoB})", "count"],
            squared=False,
        )
        cross_comparison_group[f"R ({algoB})", "percent"] = (
            cross_comparison_group[f"R ({algoB})", "count"]
            / cross_comparison_group[f"R ({algoB})", "count"].sum()
        )
        universal_metric_list.append(universal_metric)
    return universal_metric_list


def power_forgetting_curve(t, s):
    return (1 + t / (9 * s)) ** -1


class FSRS:
    def __init__(self, w: List[float]):
        self.w = w
        self.lr = 1e-2

    def init_stability(self, rating):
        return self.w[rating - 1]

    def init_difficulty(self, rating):
        return max(1, min(self.w[4] - self.w[5] * (rating - 3), 10))

    def next_difficulty(self, last_d, rating):
        return max(1, min(last_d - self.w[6] * (rating - 3), 10))

    def stability_after_success(self, last_s, last_d, r, rating):
        hard_penalty = self.w[15] if rating == 2 else 1
        easy_bonus = self.w[16] if rating == 4 else 1
        new_s = last_s * (
            1
            + math.exp(self.w[8])
            * (11 - last_d)
            * math.pow(last_s, -self.w[9])
            * (math.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return max(0.1, min(new_s, 36500))

    def stability_after_failure(self, last_s, last_d, r):
        new_s = (
            self.w[11]
            * math.pow(last_d, -self.w[12])
            * (math.pow(last_s + 1, self.w[13]) - 1)
            * math.exp((1 - r) * self.w[14])
        )
        return max(0.1, min(new_s, 36500))

    def clamp_weights(self):
        self.w[0] = max(0.1, min(self.w[0], 365))
        self.w[1] = max(0.1, min(self.w[1], 365))
        self.w[2] = max(0.1, min(self.w[2], 365))
        self.w[3] = max(0.1, min(self.w[3], 365))
        self.w[4] = max(1, min(self.w[4], 10))
        self.w[5] = max(0.1, min(self.w[5], 5))
        self.w[6] = max(0.1, min(self.w[6], 5))
        self.w[7] = max(0, min(self.w[7], 0.5))
        self.w[8] = max(0, min(self.w[8], 3))
        self.w[9] = max(0.1, min(self.w[9], 0.8))
        self.w[10] = max(0.01, min(self.w[10], 2.5))
        self.w[11] = max(0.5, min(self.w[11], 5))
        self.w[12] = max(0.01, min(self.w[12], 0.2))
        self.w[13] = max(0.01, min(self.w[13], 0.9))
        self.w[14] = max(0.01, min(self.w[14], 2))
        self.w[15] = max(0, min(self.w[15], 1))
        self.w[16] = max(1, min(self.w[16], 4))

    def update_weights(self, last_s, last_d, cur_s, last_rating, delta_t, y):
        if last_s is None:
            grad_s = (-delta_t * y + 9 * self.w[last_rating - 1] * (1 - y)) / (
                self.w[last_rating - 1] * (delta_t + 9 * self.w[last_rating - 1])
            )
            self.w[last_rating - 1] -= grad_s * self.lr * 10
        elif last_rating > 1:
            # (9*s*(1 - y) - t*y)/(s*(9*s + t)), (9*s*(1 - y) - t*y)/(s*(9*s + t))
            grad_s = (9 * cur_s * (1 - y) - delta_t * y) / (
                cur_s * (9 * cur_s + delta_t)
            )
            # last_s**(1 - w9)*w15*w16*(1 - exp(t*w10/(9*s + t)))*(last_d - 11)*exp(w8)
            grad_8 = (
                last_s ** (1 - self.w[9])
                * (self.w[15] if last_rating == 2 else 1)
                * (self.w[16] if last_rating == 4 else 1)
                * (1 - math.exp(delta_t * self.w[10] / (9 * last_s + delta_t)))
                * (last_d - 11)
                * math.exp(self.w[8])
            )
            # last_s**(1 - w9)*w15*w16*(last_d - 11)*(exp(t*w10/(9*s + t)) - 1)*exp(w8)*log(last_s)
            grad_9 = (
                last_s ** (1 - self.w[9])
                * (self.w[15] if last_rating == 2 else 1)
                * (self.w[16] if last_rating == 4 else 1)
                * (last_d - 11)
                * (math.exp(delta_t * self.w[10] / (9 * last_s + delta_t)) - 1)
                * math.exp(self.w[8])
                * math.log(last_s)
            )
            # last_s**(1 - w9)*t*w15*w16*(11 - last_d)*exp(t*w10/(9*s + t) + w8)/(9*s + t)
            grad_10 = (
                last_s ** (1 - self.w[9])
                * delta_t
                * (self.w[15] if last_rating == 2 else 1)
                * (self.w[16] if last_rating == 4 else 1)
                * (11 - last_d)
                * math.exp((delta_t * self.w[10] / (9 * last_s + delta_t)) + self.w[8])
                / (9 * last_s + delta_t)
            )
            # last_s**(1 - w9)*w16*(1 - exp(t*w10/(9*last_s + t)))*(last_d - 11)*exp(w8)
            grad_15 = (
                (
                    last_s ** (1 - self.w[9])
                    * (self.w[16] if last_rating == 4 else 1)
                    * (1 - math.exp(delta_t * self.w[10] / (9 * last_s + delta_t)))
                    * (last_d - 11)
                    * math.exp(self.w[8])
                )
                if last_rating == 2
                else 0
            )
            # last_s**(1 - w9)*w15*(1 - exp(t*w10/(9*last_s + t)))*(last_d - 11)*exp(w8)
            grad_16 = (
                (
                    last_s ** (1 - self.w[9])
                    * (self.w[15] if last_rating == 4 else 1)
                    * (1 - math.exp(delta_t * self.w[10] / (9 * last_s + delta_t)))
                    * (last_d - 11)
                    * math.exp(self.w[8])
                )
                if last_rating == 2
                else 0
            )
            # last_s**(1 - w9)*w15*w16*(1 - exp(t*w10/(9*last_s + t)))*exp(w8))
            grad_last_d = (
                last_s ** (1 - self.w[9])
                * (self.w[15] if last_rating == 2 else 1)
                * (self.w[16] if last_rating == 4 else 1)
                * (1 - math.exp(delta_t * self.w[10] / (9 * last_s + delta_t)))
                * math.exp(self.w[8])
            )
            grad_w4 = 1
            grad_w5 = 3 - last_rating
            grad_w6 = 3 - last_rating
            self.w[4] -= grad_s * grad_last_d * grad_w4 * self.lr
            self.w[5] -= grad_s * grad_last_d * grad_w5 * self.lr
            self.w[6] -= grad_s * grad_last_d * grad_w6 * self.lr
            self.w[8] -= grad_s * grad_8 * self.lr
            self.w[9] -= grad_s * grad_9 * self.lr
            self.w[10] -= grad_s * grad_10 * self.lr
            self.w[15] -= grad_s * grad_15 * self.lr
            self.w[16] -= grad_s * grad_16 * self.lr
        else:
            # (9*s*(1 - y) - t*y)/(s*(9*s + t)), (9*s*(1 - y) - t*y)/(s*(9*s + t))
            grad_s = (9 * cur_s * (1 - y) - delta_t * y) / (
                cur_s * (9 * cur_s + delta_t)
            )
            # ((s + 1)**w13 - 1)*exp(t*w14/(9*s + t))/d**w12
            grad_11 = (
                ((last_s + 1) ** self.w[13] - 1)
                * math.exp(delta_t * self.w[14] / (9 * last_s + delta_t))
                / last_d ** self.w[12]
            )
            #  w11*(1 - (s + 1)**w13)*exp(t*w14/(9*s + t))*log(d)/d**w12
            grad_12 = (
                self.w[11]
                * (1 - (last_s + 1) ** self.w[13])
                * math.exp(delta_t * self.w[14] / (9 * last_s + delta_t))
                * math.log(last_d)
                / last_d ** self.w[12]
            )
            # w11*(s + 1)**w13*exp(t*w14/(9*s + t))*log(s + 1)/d**w12
            grad_13 = (
                self.w[11]
                * (last_s + 1) ** self.w[13]
                * math.exp(delta_t * self.w[14] / (9 * last_s + delta_t))
                * math.log(last_s + 1)
                / last_d ** self.w[12]
            )
            # t*w11*((s + 1)**w13 - 1)*exp(t*w14/(9*s + t))/(d**w12*(9*s + t))
            grad_14 = (
                delta_t
                * self.w[11]
                * ((last_s + 1) ** self.w[13] - 1)
                * math.exp(delta_t * self.w[14] / (9 * last_s + delta_t))
                / (last_d ** self.w[12] * (9 * last_s + delta_t))
            )
            # last_d**(-w12 - 1)*w11*w12*(1 - (last_s + 1)**w13)*exp(t*w14/(9*last_s + t)))
            grad_last_d = (
                last_d ** (-self.w[12] - 1)
                * self.w[11]
                * self.w[12]
                * (1 - (last_s + 1) ** self.w[13])
                * math.exp(delta_t * self.w[14] / (9 * last_s + delta_t))
            )
            grad_w4 = 1
            grad_w5 = 3 - last_rating
            grad_w6 = 3 - last_rating
            self.w[4] -= grad_s * grad_last_d * grad_w4 * self.lr
            self.w[5] -= grad_s * grad_last_d * grad_w5 * self.lr
            self.w[6] -= grad_s * grad_last_d * grad_w6 * self.lr
            self.w[11] -= grad_s * grad_11 * self.lr
            self.w[12] -= grad_s * grad_12 * self.lr
            self.w[13] -= grad_s * grad_13 * self.lr
            self.w[14] -= grad_s * grad_14 * self.lr

        self.clamp_weights()

    def step(self, delta_t, rating, last_s, last_d):
        if last_s is None:
            return self.init_stability(rating), self.init_difficulty(rating)
        else:
            r = power_forgetting_curve(delta_t, last_s)
            if rating == 1:
                return self.stability_after_failure(
                    last_s, last_d, r
                ), self.next_difficulty(last_d, rating)
            else:
                return self.stability_after_success(
                    last_s, last_d, r, rating
                ), self.next_difficulty(last_d, rating)

    def forward(self, inputs):
        last_s = None
        last_d = None
        outputs = []
        for delta_t, rating in inputs:
            last_s, last_d = self.step(delta_t, rating, last_s, last_d)
            outputs.append((last_s, last_d))

        self.last_s, self.last_d = outputs[-2] if len(outputs) > 1 else (None, None)
        self.new_s, self.new_d = outputs[-1]
        self.last_rating = inputs[-1][1]
        return outputs

    def backward(self, delta_t, y):
        self.update_weights(
            self.last_s, self.last_d, self.new_s, self.last_rating, delta_t, y
        )


def cum_concat(x):
    return list(accumulate(x))


def create_time_series(df):
    df = df[(df["delta_t"] != 0) & (df["rating"].isin([1, 2, 3, 4]))].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    df["inputs"] = [
        tuple(zip(t_item[:-1], r_item[:-1]))
        for t_sublist, r_sublist in zip(t_history, r_history)
        for t_item, r_item in zip(t_sublist, r_sublist)
    ]
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    return df[df["delta_t"] > 0].sort_values(by=["review_th"]).reset_index(drop=True)


if __name__ == "__main__":
    path = "../fsrs-benchmark/dataset/1.csv"
    df = pd.read_csv(path)
    df = create_time_series(df)
    init_w = [
        0.4,
        0.9,
        2.3,
        10.9,
        4.93,
        0.94,
        0.86,
        0.01,
        1.49,
        0.14,
        0.94,
        2.18,
        0.05,
        0.34,
        1.26,
        0.29,
        2.61,
    ]
    fsrs = FSRS(init_w)
    for n in range(5):
        y_list = []
        p_list = []
        for index in df.index:
            sample = df.loc[index]
            outputs = fsrs.forward(sample["inputs"])
            new_s, new_d = outputs[-1]
            delta_t, y = sample["delta_t"], sample["y"]
            p = power_forgetting_curve(delta_t, new_s)
            fsrs.backward(delta_t, y)
            y_list.append(y)
            p_list.append(p)

        print(fsrs.w)
        print(f"RMSE: {mean_squared_error(y_list, p_list, squared=False):.4f}")
        print(f"LogLoss: {log_loss(y_list, p_list):.4f}")
        print(
            f"""RMSE(bins): {cross_comparison(pd.DataFrame({"y": y_list, "R (FSRS)": p_list}), "FSRS", "FSRS")[0]:.4f}"""
        )
