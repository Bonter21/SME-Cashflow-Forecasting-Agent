import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


def analyze_cashflow(file_path, prediction_days=30):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.lower().str.strip()

    date_cols = ["date", "transaction_date", "trans_date", "timestamp"]
    amount_cols = ["amount", "balance", "value", "total"]

    date_col = next(
        (c for c in df.columns if any(d in c for d in date_cols)), df.columns[0]
    )
    amount_col = next(
        (c for c in df.columns if any(a in c for a in amount_cols)), df.columns[1]
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[amount_col])

    df = df.sort_values(date_col)
    df["running_balance"] = df[amount_col].cumsum()

    daily_df = (
        df.groupby(df[date_col].dt.date)
        .agg({amount_col: "sum", "running_balance": "last"})
        .reset_index()
    )
    daily_df.columns = ["date", "daily_change", "balance"]
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    current_balance = df["running_balance"].iloc[-1]

    daily_df = daily_df.copy()
    daily_df["day_num"] = (daily_df["date"] - daily_df["date"].min()).dt.days

    X = daily_df["day_num"].values.reshape(-1, 1)
    y = daily_df["daily_change"].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = daily_df["date"].max()
    last_day = daily_df["day_num"].max()

    future_dates = [
        last_date + timedelta(days=i) for i in range(1, prediction_days + 1)
    ]
    future_day_nums = [last_day + i for i in range(1, prediction_days + 1)]
    future_X = np.array(future_day_nums).reshape(-1, 1)
    predictions = model.predict(future_X)

    last_balance = daily_df["balance"].iloc[-1]
    predicted_balances = [
        last_balance + sum(predictions[: i + 1]) for i in range(len(predictions))
    ]

    balance_change = predicted_balances[-1] - current_balance

    return {
        "current_balance": current_balance,
        "predicted_balance": predicted_balances[-1],
        "balance_change": balance_change,
        "min_predicted": min(predicted_balances),
        "max_predicted": max(predicted_balances),
        "avg_daily": model.coef_[0],
        "trend": "up" if model.coef_[0] > 0 else "down",
        "dates": future_dates,
        "predictions": predictions.tolist(),
        "balances": predicted_balances,
    }


def generate_conclusion_text(results, prediction_days=30):
    cb = results["current_balance"]
    pb = results["predicted_balance"]
    bc = results["balance_change"]
    min_p = results["min_predicted"]
    max_p = results["max_predicted"]
    trend = results["trend"]
    avg = results["avg_daily"]

    change_pct = (bc / abs(cb)) * 100 if cb != 0 else 0

    if trend == "up" and bc > 0:
        if change_pct > 20:
            status = "EXCELLENT"
            title = "Excellent Cashflow Outlook"
            msg = f"Your cashflow is projected to grow by ${bc:,.2f} ({change_pct:.1f}%) over {prediction_days} days."
            recs = [
                "Consider investing excess cash in growth opportunities",
                "Build additional cash reserves",
                "Explore expansion opportunities",
            ]
        else:
            status = "GOOD"
            title = "Positive Cashflow Trend"
            msg = f"Balance is projected to grow by ${bc:,.2f} ({change_pct:.1f}%) over {prediction_days} days."
            recs = ["Continue current business practices", "Monitor trends monthly"]
    elif trend == "down" and bc < 0:
        if abs(change_pct) > 20 or min_p < 0:
            status = "CRITICAL"
            title = "Critical Cashflow Alert"
            msg = f"WARNING: Balance may drop by ${abs(bc):,.2f} and could go as low as ${min_p:,.2f}."
            recs = [
                "URGENT: Reduce non-essential expenses",
                "Accelerate accounts receivable collection",
                "Consider short-term financing",
            ]
        else:
            status = "WARNING"
            title = "Declining Cashflow"
            msg = f"Balance is expected to decrease by ${abs(bc):,.2f}."
            recs = ["Review expenses for cuts", "Follow up on invoices"]
    else:
        status = "STABLE"
        title = "Stable Cashflow"
        msg = f"Cashflow expected to remain stable with minimal change of ${bc:,.2f}."
        recs = ["Maintain current practices", "Look for growth opportunities"]

    lines = [
        "=" * 60,
        "CASHFLOW PREDICTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Prediction Period: {prediction_days} days",
        "-" * 60,
        "",
        "FINANCIAL SUMMARY",
        "-" * 60,
        f"Current Balance:     ${cb:>12,.2f}",
        f"Predicted Balance:   ${pb:>12,.2f}",
        f"Change:              ${bc:>12,.2f}",
        f"Lowest Predicted:    ${min_p:>12,.2f}",
        f"Highest Predicted:   ${max_p:>12,.2f}",
        f"Daily Trend:        ${avg:>12,.2f} ({trend})",
        "",
        "=" * 60,
        f"STATUS: {status}",
        "=" * 60,
        "",
        title,
        "-" * 60,
        msg,
        "",
        "RECOMMENDATIONS:",
    ]
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")

    lines.extend(
        [
            "",
            "-" * 60,
            "Generated by CashFlow Predictor AI",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <data_file.csv> [prediction_days]")
        print("Example: python generate_report.py sample_data.csv 30")
        sys.exit(1)

    file_path = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Analyzing: {file_path}")
    results = analyze_cashflow(file_path, days)

    report = generate_conclusion_text(results, days)

    output_file = "cashflow_report.txt"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport generated: {output_file}")
    print("\n" + report)
