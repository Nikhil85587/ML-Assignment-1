import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline

ROLL_NO = "IMT2024070"
TRAIN_VAR1 = f"{ROLL_NO}_train_var1.csv"
TEST_VAR1 = f"{ROLL_NO}_test_var1.csv"
TRAIN_VAR2 = f"{ROLL_NO}_train_var2.csv"
TEST_VAR2 = f"{ROLL_NO}_test_var2.csv"

N_SPLITS = 5
RANDOM_STATE = 42

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def create_model(degree, regularization, alpha):
    if regularization == "L1":
        regression_model = Lasso(alpha=alpha, max_iter=100000)
    else:
        regression_model = Ridge(alpha=alpha)

    return Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(degree=degree, include_bias=False)
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regularized_regression",
            regression_model
        )
    ])


def evaluate_degrees(X, y, degrees, problem_name):
    print("\n")
    print("=" * 80)
    print(f"{problem_name} - POLYNOMIAL DEGREE AND REGULARIZATION SEARCH")
    print("=" * 80)
    print(f"Number of samples : {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"CV folds          : {N_SPLITS}")
    print(f"Degrees tested    : "f"{min(degrees)} to {max(degrees)}")
    print(f"Alpha values      : {ALPHAS}")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    print("\n")

    print(
        f"{'Degree':<10}"
        f"{'Type':<10}"
        f"{'Alpha':<12}"
        f"{'Mean MSE':<18}"
        f"{'Std MSE':<18}"
        f"{'Mean R2':<18}"
        f"{'Std R2':<18}"
    )

    print("-" * 100)

    for degree in degrees:
        for regularization in ["L1", "L2"]:
            for alpha in ALPHAS:

                print(
                    f"Testing degree {degree}, "
                    f"{regularization}, alpha={alpha}...",
                    end=" ",
                    flush=True
                )

                model = create_model(
                    degree,
                    regularization,
                    alpha
                )

                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=kfold,
                    scoring={
                        "mse": "neg_mean_squared_error",
                        "r2": "r2"
                    },
                    n_jobs=-1
                )

                mse_scores = -scores["test_mse"]
                r2_scores = scores["test_r2"]

                mean_mse = np.mean(mse_scores)
                std_mse = np.std(mse_scores)

                mean_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores)

                results.append({
                    "degree": degree,
                    "regularization": regularization,
                    "alpha": alpha,
                    "mean_mse": mean_mse,
                    "std_mse": std_mse,
                    "mean_r2": mean_r2,
                    "std_r2": std_r2
                })

                print("done")

                print(
                    f"{degree:<10}"
                    f"{regularization:<10}"
                    f"{alpha:<12}"
                    f"{mean_mse:<18.8f}"
                    f"{std_mse:<18.8f}"
                    f"{mean_r2:<18.8f}"
                    f"{std_r2:<18.8f}"
                )

    results_df = pd.DataFrame(results)

    best_index = results_df["mean_mse"].idxmin()

    best_row = results_df.loc[best_index]

    best_degree = int(best_row["degree"])
    best_regularization = best_row["regularization"]
    best_alpha = float(best_row["alpha"])

    print("\n")
    print("-" * 80)
    print(f"BEST MODEL FOR {problem_name}")
    print("-" * 80)

    print(f"Degree          : {best_degree}")
    print(f"Regularization  : {best_regularization}")
    print(f"Alpha           : {best_alpha}")
    print(f"Mean CV MSE     : {best_row['mean_mse']:.8f}")
    print(f"Std CV MSE      : {best_row['std_mse']:.8f}")
    print(f"Mean CV R2      : {best_row['mean_r2']:.8f}")
    print(f"Std CV R2       : {best_row['std_r2']:.8f}")

    return (
        results_df,
        best_degree,
        best_regularization,
        best_alpha
    )


def train_final_model(X, y, degree, regularization, alpha):
    model = create_model(
        degree,
        regularization,
        alpha
    )

    model.fit(X, y)

    return model


def plot_results(results_df, problem_name, best_degree, best_regularization, best_alpha, filename):

    plot_df = results_df[
        (results_df["regularization"] == best_regularization) &
        (results_df["alpha"] == best_alpha)
    ]

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        plot_df["degree"],
        plot_df["mean_mse"],
        yerr=plot_df["std_mse"],
        marker="o",
        capsize=4
    )

    best_row = plot_df[
        plot_df["degree"] == best_degree
    ].iloc[0]

    plt.scatter(
        best_degree,
        best_row["mean_mse"],
        s=120,
        zorder=5,
        label=f"Selected degree = {best_degree}"
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("5-Fold Validation MSE")

    plt.title(
        f"{problem_name}: Validation MSE vs Polynomial Degree\n"
        f"{best_regularization}, alpha={best_alpha}"
    )

    plt.xticks(plot_df["degree"])

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\n")
print("#" * 80)
print("# VAR1 - STEAM TURBINE OPTIMIZATION")
print("#" * 80)

train1 = pd.read_csv(TRAIN_VAR1)
test1 = pd.read_csv(TEST_VAR1)

print("\nTraining shape:", train1.shape)
print("Test shape    :", test1.shape)

print("\nTraining columns:")
print(train1.columns.tolist())

features_var1 = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6"
]

X1 = train1[features_var1]
y1 = train1["y"]

X1_test = test1[features_var1]

print("\nVAR1 features:")
print(features_var1)

(
    results_var1,
    best_degree_var1,
    best_regularization_var1,
    best_alpha_var1
) = evaluate_degrees(
    X1,
    y1,
    degrees=range(1, 11),
    problem_name="VAR1"
)

results_var1.to_csv(
    f"{ROLL_NO}_var1_degree_results.csv",
    index=False
)

plot_results(
    results_var1,
    "VAR1",
    best_degree_var1,
    best_regularization_var1,
    best_alpha_var1,
    f"{ROLL_NO}_var1_degree_selection.png"
)

print("\n")
print("=" * 80)
print("VAR1 - FINAL TRAINING")
print("=" * 80)

print(f"Features       : {features_var1}")
print(f"Selected degree: {best_degree_var1}")
print(f"Regularization : {best_regularization_var1}")
print(f"Alpha          : {best_alpha_var1}")
print("Training data  : 100%")

final_model_var1 = train_final_model(
    X1,
    y1,
    best_degree_var1,
    best_regularization_var1,
    best_alpha_var1
)

pred_var1 = final_model_var1.predict(X1_test)

output_var1 = pd.DataFrame({
    "y": pred_var1
})

output_var1.to_csv(
    f"{ROLL_NO}_pred_var1.csv",
    index=False
)

print(f"\nCreated: {ROLL_NO}_pred_var1.csv")


print("\n")
print("#" * 80)
print("# VAR2 - SUBTERRANEAN THERMAL RESERVOIR MAPPING")
print("#" * 80)

train2 = pd.read_csv(TRAIN_VAR2)
test2 = pd.read_csv(TEST_VAR2)

print("\nTraining shape:", train2.shape)
print("Test shape    :", test2.shape)

print("\nTraining columns:")
print(train2.columns.tolist())

features_var2 = [
    "x1",
    "x2",
    "x3"
]

X2 = train2[features_var2]
y2 = train2["y"]

X2_test = test2[features_var2]

print("\nVAR2 features:")
print(features_var2)

(
    results_var2,
    best_degree_var2,
    best_regularization_var2,
    best_alpha_var2
) = evaluate_degrees(
    X2,
    y2,
    degrees=range(1, 21),
    problem_name="VAR2"
)

results_var2.to_csv(
    f"{ROLL_NO}_var2_degree_results.csv",
    index=False
)

plot_results(
    results_var2,
    "VAR2",
    best_degree_var2,
    best_regularization_var2,
    best_alpha_var2,
    f"{ROLL_NO}_var2_degree_selection.png"
)

print("\n")
print("=" * 80)
print("VAR2 - FINAL TRAINING")
print("=" * 80)

print(f"Features       : {features_var2}")
print(f"Selected degree: {best_degree_var2}")
print(f"Regularization : {best_regularization_var2}")
print(f"Alpha          : {best_alpha_var2}")
print("Training data  : 100%")

final_model_var2 = train_final_model(
    X2,
    y2,
    best_degree_var2,
    best_regularization_var2,
    best_alpha_var2
)

pred_var2 = final_model_var2.predict(X2_test)

output_var2 = pd.DataFrame({
    "y": pred_var2
})

output_var2.to_csv(
    f"{ROLL_NO}_pred_var2.csv",
    index=False
)

print(f"\nCreated: {ROLL_NO}_pred_var2.csv")


print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

var1_best = results_var1[
    (results_var1["degree"] == best_degree_var1) &
    (results_var1["regularization"] == best_regularization_var1) &
    (results_var1["alpha"] == best_alpha_var1)
].iloc[0]

print("\nVAR1")
print("-" * 60)

print(f"Features       : {features_var1}")
print(f"Selected degree: {best_degree_var1}")
print(f"Regularization : {best_regularization_var1}")
print(f"Alpha          : {best_alpha_var1}")
print(f"Mean CV MSE    : "f"{var1_best['mean_mse']:.8f}")
print(f"Std CV MSE     : "f"{var1_best['std_mse']:.8f}")
print(f"Mean CV R2     : "f"{var1_best['mean_r2']:.8f}")
print(f"Std CV R2      : "f"{var1_best['std_r2']:.8f}")


var2_best = results_var2[
    (results_var2["degree"] == best_degree_var2) &
    (results_var2["regularization"] == best_regularization_var2) &
    (results_var2["alpha"] == best_alpha_var2)
].iloc[0]

print("\nVAR2")
print("-" * 60)

print(f"Features       : {features_var2}")
print(f"Selected degree: {best_degree_var2}")
print(f"Regularization : {best_regularization_var2}")
print(f"Alpha          : {best_alpha_var2}")
print(f"Mean CV MSE    : "f"{var2_best['mean_mse']:.8f}")
print(f"Std CV MSE     : "f"{var2_best['std_mse']:.8f}")
print(f"Mean CV R2     : "f"{var2_best['mean_r2']:.8f}")
print(f"Std CV R2      : "f"{var2_best['std_r2']:.8f}")


print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print("\nPrediction files:")
print(f"  {ROLL_NO}_pred_var1.csv")
print(f"  {ROLL_NO}_pred_var2.csv")

print("\nCV result files:")
print(f"  {ROLL_NO}_var1_degree_results.csv")
print(f"  {ROLL_NO}_var2_degree_results.csv")

print("\nPlots:")
print(f"  {ROLL_NO}_var1_degree_selection.png")
print(f"  {ROLL_NO}_var2_degree_selection.png")

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)