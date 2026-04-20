import pandas as pd
import csv
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

CSV_FILE = "finance_data.csv"
COLUMNS  = ["date", "amount", "category", "description"]
FORMAT   = "%d-%m-%Y"


def initialize_csv():
    try:
        pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)


def parse_date_flexible(d):
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(d).strip(), fmt)
        except Exception:
            continue
    return None


def add_entry(date, amount, category, description):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow({"date": date, "amount": amount,
                         "category": category, "description": description})


def get_transactions(start_date=None, end_date=None):
    try:
        df = pd.read_csv(CSV_FILE, dtype=str)
        df.columns = df.columns.str.strip()
        if df.empty or "amount" not in df.columns:
            return pd.DataFrame(columns=COLUMNS)

        df["amount"]      = pd.to_numeric(df["amount"], errors="coerce")
        df["category"]    = df["category"].str.strip()
        df["description"] = df["description"].fillna("").str.strip()
        df = df.dropna(subset=["amount"])

        df["date_parsed"] = df["date"].apply(parse_date_flexible)
        df = df.dropna(subset=["date_parsed"])

        if start_date and end_date:
            start = parse_date_flexible(start_date)
            end   = parse_date_flexible(end_date)
            if start and end:
                df = df[(df["date_parsed"] >= start) & (df["date_parsed"] <= end)]

        df["date"] = df["date_parsed"].dt.strftime(FORMAT)
        df = df.drop(columns=["date_parsed"]).sort_values("date")
        return df.reset_index(drop=True)
    except Exception as e:
        print("ERROR get_transactions:", e)
        return pd.DataFrame(columns=COLUMNS)


@app.route("/")
def dashboard():
    initialize_csv()
    return render_template("dashboard.html")


@app.route("/api/transactions")
def api_transactions():
    start = request.args.get("start")
    end   = request.args.get("end")
    df    = get_transactions(start, end)

    empty = {"transactions": [], "total_income": 0, "total_expense": 0,
             "net_savings": 0, "chart_dates": [], "chart_income": [],
             "chart_expense": [], "category_data": [0, 0],
             "monthly_income": [], "monthly_expense": [], "monthly_labels": []}

    if df.empty:
        return jsonify(empty)

    total_income  = float(df[df["category"] == "Income"]["amount"].sum())
    total_expense = float(df[df["category"] == "Expense"]["amount"].sum())
    net_savings   = total_income - total_expense

    df_chart = df.copy()
    df_chart["date_parsed"] = pd.to_datetime(df_chart["date"], format=FORMAT)
    df_chart = df_chart.sort_values("date_parsed")

    income_by_date  = df_chart[df_chart["category"] == "Income"].groupby("date")["amount"].sum()
    expense_by_date = df_chart[df_chart["category"] == "Expense"].groupby("date")["amount"].sum()
    all_dates       = sorted(df_chart["date"].unique())

    df_chart["month"] = df_chart["date_parsed"].dt.strftime("%b %Y")
    monthly_income    = df_chart[df_chart["category"] == "Income"].groupby("month")["amount"].sum()
    monthly_expense   = df_chart[df_chart["category"] == "Expense"].groupby("month")["amount"].sum()
    monthly_labels    = list(dict.fromkeys(df_chart["month"].tolist()))

    return jsonify({
        "transactions":   df.to_dict(orient="records"),
        "total_income":   total_income,
        "total_expense":  total_expense,
        "net_savings":    net_savings,
        "chart_dates":    list(all_dates),
        "chart_income":   [float(income_by_date.get(d, 0)) for d in all_dates],
        "chart_expense":  [float(expense_by_date.get(d, 0)) for d in all_dates],
        "category_data":  [total_income, total_expense],
        "monthly_labels": monthly_labels,
        "monthly_income": [float(monthly_income.get(m, 0)) for m in monthly_labels],
        "monthly_expense":[float(monthly_expense.get(m, 0)) for m in monthly_labels],
    })


@app.route("/api/add", methods=["POST"])
def api_add():
    data = request.json
    try:
        add_entry(data["date"], float(data["amount"]),
                  data["category"], data["description"])
        return jsonify({"success": True, "message": "Transaction added!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/run_ml", methods=["POST"])
def api_run_ml():
    try:
        result = subprocess.run(
            ["python", "ml_analysis.py"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return jsonify({"success": True,
                            "message": "ML analysis complete! Refresh Power BI now.",
                            "output": result.stdout})
        else:
            return jsonify({"success": False,
                            "message": "ML script error",
                            "output": result.stderr})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    initialize_csv()
    app.run(debug=True)