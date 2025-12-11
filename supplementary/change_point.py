# changepoint_models.py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Tuple

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_three_param_cp(
        temp: np.ndarray,
        kwh: np.ndarray,
        Tmin: float,
        Tmax: float,
        step: float = 1.0,
        mode: str = "auto",  # NEW: "heating", "cooling", or "auto"
) -> Dict[str, Any]:
    """
    Fit 3-parameter change-point model.

    mode:
        "heating": use heating model only
        "cooling": use cooling model only
        "auto":    evaluate both and pick lowest RMSE

    Heating CP:  kWh = b0 + b1 * max(0, Tb - T)
    Cooling CP:  kWh = b0 + b1 * max(0, T - Tb)
    """

    mode = mode.lower()
    if mode not in ("heating", "cooling", "auto"):
        raise ValueError("mode must be 'heating', 'cooling', or 'auto'")

    best = {"rmse": np.inf}
    candidates = np.arange(Tmin, Tmax + step / 2, step)

    for Tb in candidates:

        # ----------------------------
        # Cooling Model
        # ----------------------------
        if mode in ("cooling", "auto"):
            X_cool = np.maximum(0.0, temp - Tb).reshape(-1, 1)
            mdl = LinearRegression().fit(X_cool, kwh)
            pred = mdl.predict(X_cool)
            rmse_val = rmse(kwh, pred)

            if rmse_val < best["rmse"]:
                best = {
                    "mode": "cooling",
                    "Tb": float(Tb),
                    "model": mdl,
                    "pred": pred,
                    "rmse": rmse_val,
                    "r2": mdl.score(X_cool, kwh),
                }

        # ----------------------------
        # Heating Model
        # ----------------------------
        if mode in ("heating", "auto"):
            X_heat = np.maximum(0.0, Tb - temp).reshape(-1, 1)
            mdl = LinearRegression().fit(X_heat, kwh)
            pred = mdl.predict(X_heat)
            rmse_val = rmse(kwh, pred)

            if rmse_val < best["rmse"]:
                best = {
                    "mode": "heating",
                    "Tb": float(Tb),
                    "model": mdl,
                    "pred": pred,
                    "rmse": rmse_val,
                    "r2": mdl.score(X_heat, kwh),
                }

    return best


def fit_five_param_deadband(
    temp: np.ndarray,
    kwh: np.ndarray,
    Tmin: float,
    Tmax: float,
    step: float = 1.0
) -> Dict[str, Any]:
    """
    Fit 5-parameter deadband model:
      kwh = beta0 + beta_h * max(0, Tb_low - T) + beta_c * max(0, T - Tb_high)
    by grid-searching Tb_low, Tb_high (Tb_low < Tb_high).
    Returns dict with Tb_low, Tb_high, model, pred, rmse, r2.
    """
    best = {"rmse": np.inf}
    candidates = np.arange(Tmin, Tmax + step/2, step)
    for Tb_low in candidates:
        for Tb_high in candidates:
            if Tb_high <= Tb_low:
                continue
            heat = np.maximum(0.0, Tb_low - temp)
            cool = np.maximum(0.0, temp - Tb_high)
            X = np.column_stack([heat, cool])
            model = LinearRegression().fit(X, kwh)
            pred = model.predict(X)
            r = rmse(kwh, pred)
            r2 = model.score(X, kwh)
            if r < best["rmse"]:
                best = {
                    "Tb_low": float(Tb_low),
                    "Tb_high": float(Tb_high),
                    "model": model,
                    "pred": pred,
                    "rmse": r,
                    "r2": float(r2)
                }
    return best

def select_model_by_rmse_r2(
    three_res: Dict[str, Any],
    five_res: Dict[str, Any],
    rel_tol_pct: float,
    mean_kwh: float
) -> Tuple[str, Dict[str, Any]]:
    """
    Select preferred model using RMSE (primary) and R2 (tiebreaker).
    rel_tol_pct: relative tolerance percent (e.g., 0.1 means 0.1% of mean_kwh).
    Returns (preferred_label, chosen_result_dict).
    """
    tol_abs = (rel_tol_pct / 100.0) * mean_kwh
    rmse3 = three_res["rmse"]
    rmse5 = five_res["rmse"]
    r23 = three_res["r2"]
    r25 = five_res["r2"]

    if rmse3 + tol_abs < rmse5:
        return "3-parameter", three_res
    elif rmse5 + tol_abs < rmse3:
        return "5-parameter", five_res
    else:
        # tie -> pick higher R2
        if r23 >= r25:
            return "3-parameter", three_res
        else:
            return "5-parameter", five_res


def predict_3p_for_plot(
        T_plot: np.ndarray,
        Tb: float,
        model: LinearRegression,
        mode: str = None,  # "heating", "cooling", or None for auto
) -> np.ndarray:
    """
    Return predictions for a temperature array using a 3-parameter CP model.

    Parameters:
        T_plot : np.ndarray
            Array of temperatures for prediction.
        Tb : float
            Balance point temperature.
        model : LinearRegression
            Fitted 3-parameter LinearRegression model.
        mode : str, optional
            "heating" or "cooling". If None, defaults to "cooling".

    Returns:
        np.ndarray
            Predicted energy values.
    """
    # Default to cooling if mode is not provided
    if mode is None:
        mode = "cooling"

    mode = mode.lower()

    if mode == "cooling":
        # Cooling model: max(0, T - Tb)
        X = np.maximum(0.0, T_plot - Tb).reshape(-1, 1)
    elif mode == "heating":
        # Heating model: max(0, Tb - T)
        X = np.maximum(0.0, Tb - T_plot).reshape(-1, 1)
    else:
        raise ValueError("mode must be 'heating' or 'cooling'")

    return model.predict(X)


def predict_5p_for_plot(T_plot: np.ndarray, Tb_low: float, Tb_high: float, model: LinearRegression) -> np.ndarray:
    """Return model predictions for a temperature array using a 5p model object."""
    heat = np.maximum(0.0, Tb_low - T_plot)
    cool = np.maximum(0.0, T_plot - Tb_high)
    X = np.column_stack([heat, cool])
    return model.predict(X)
