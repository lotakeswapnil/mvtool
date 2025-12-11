# changepoint_models.py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Tuple

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def fit_three_param_cp_auto(
    temp: np.ndarray,
    kwh: np.ndarray,
    Tmin: float,
    Tmax: float,
    step: float = 1.0
) -> Dict[str, Any]:
    """
    Fit BOTH heating and cooling 3-parameter change-point (CP) models:

    Cooling CP:  kWh = b0 + b1 * max(0, T - Tb)
    Heating CP:  kWh = b0 + b1 * max(0, Tb - T)

    Grid-search Tb in [Tmin, Tmax], choose the model with lowest RMSE.

    Returns:
      {
        "mode": "heating" or "cooling",
        "Tb": balance point,
        "model": sklearn model,
        "pred": predictions,
        "rmse": RMSE,
        "r2": R²
      }
    """
    best = {"rmse": np.inf}
    candidates = np.arange(Tmin, Tmax + step/2, step)

    for Tb in candidates:
        # --- Cooling model ----------------------------------------------------
        X_cool = np.maximum(0.0, temp - Tb).reshape(-1, 1)
        model_cool = LinearRegression().fit(X_cool, kwh)
        pred_cool = model_cool.predict(X_cool)
        rmse_cool = rmse(kwh, pred_cool)
        r2_cool = model_cool.score(X_cool, kwh)

        if rmse_cool < best["rmse"]:
            best = {
                "mode": "cooling",
                "Tb": float(Tb),
                "model": model_cool,
                "pred": pred_cool,
                "rmse": rmse_cool,
                "r2": float(r2_cool)
            }

        # --- Heating model ----------------------------------------------------
        X_heat = np.maximum(0.0, Tb - temp).reshape(-1, 1)
        model_heat = LinearRegression().fit(X_heat, kwh)
        pred_heat = model_heat.predict(X_heat)
        rmse_heat = rmse(kwh, pred_heat)
        r2_heat = model_heat.score(X_heat, kwh)

        if rmse_heat < best["rmse"]:
            best = {
                "mode": "heating",
                "Tb": float(Tb),
                "model": model_heat,
                "pred": pred_heat,
                "rmse": rmse_heat,
                "r2": float(r2_heat)
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

def predict_3p_for_plot(T_plot: np.ndarray, Tb: float, model: LinearRegression) -> np.ndarray:
    """Return model predictions for a temperature array using a 3p model object."""
    X = np.maximum(0.0, T_plot - Tb).reshape(-1, 1)
    return model.predict(X)

def predict_5p_for_plot(T_plot: np.ndarray, Tb_low: float, Tb_high: float, model: LinearRegression) -> np.ndarray:
    """Return model predictions for a temperature array using a 5p model object."""
    heat = np.maximum(0.0, Tb_low - T_plot)
    cool = np.maximum(0.0, T_plot - Tb_high)
    X = np.column_stack([heat, cool])
    return model.predict(X)
