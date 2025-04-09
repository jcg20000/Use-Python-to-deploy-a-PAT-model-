from flask import Flask, request, render_template_string, send_file
import os
import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
import pyodbc
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)

# Paths and config
MODEL_PATH = r"D:\\Model\\pls_model_peach_brix.pkl"
DATA_DIR = r"D:\\Data"
REPORT_DIR = r"D:\\Report"
os.makedirs(REPORT_DIR, exist_ok=True)

# SQL Server connection string
conn_str = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=MJ0DF884\\SQLEXPRESS;"
    "Database=PLS_output;"
    "Trusted_Connection=yes;"
)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>PLS Web App</title></head>
<body>
  <h2>Run PLS Batch Analysis</h2>
  <form method="post">
    <label>Batch ID:</label><br>
    <input type="text" name="batch_id" required><br>
    <label>Instrument S/N:</label><br>
    <input type="text" name="instrument_sn" required><br><br>
    <input type="submit" value="Run Analysis">
  </form>
  {% if result %}
    <h3>Processed {{ result.num_files }} file(s).</h3>
    <a href="/download/{{ result.filename }}">Download PDF Report</a>
  {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        batch_id = request.form['batch_id']
        instrument_sn = request.form['instrument_sn']

        pls = joblib.load(MODEL_PATH)
        matched_files = [f for f in os.listdir(DATA_DIR) if batch_id in f and f.endswith(".csv")]
        if not matched_files:
            return render_template_string(HTML_TEMPLATE, result=None)

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='results' AND xtype='U')
            CREATE TABLE results (
                batch_id NVARCHAR(255),
                instrument_sn NVARCHAR(255),
                prediction FLOAT,
                t2 FLOAT,
                q_residual FLOAT,
                test_time NVARCHAR(255)
            )
        """)
        conn.commit()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="PLS Test Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Batch ID: {batch_id}", ln=True)
        pdf.cell(200, 10, txt=f"Instrument S/N: {instrument_sn}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", size=10)
        pdf.cell(40, 10, "Filename", 1)
        pdf.cell(30, 10, "Prediction", 1)
        pdf.cell(30, 10, "T2", 1)
        pdf.cell(30, 10, "Q Residual", 1)
        pdf.cell(60, 10, "Test Time", 1)
        pdf.ln()

        for fname in matched_files:
            path = os.path.join(DATA_DIR, fname)
            timestamp = os.path.getctime(path)
            test_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            X = pd.read_csv(path, header=None).iloc[:, 1].values.reshape(1, -1)
            X_snv = (X - np.mean(X)) / np.std(X)
            X_deriv = savgol_filter(X_snv, window_length=7, polyorder=2, deriv=1)
            X_scaled = (X_deriv - pls._x_mean) / pls._x_std
            prediction = pls.predict(X_scaled)[0]

            T_new = X_scaled @ pls.x_weights_
            T_train = pls.x_scores_
            var_t = np.var(T_train, axis=0)
            T2 = np.sum((T_new ** 2) / var_t, axis=1)[0]

            X_reconstructed = T_new @ pls.x_loadings_.T
            residual = X_scaled - X_reconstructed
            Q = np.sum(residual ** 2)

            cursor.execute("""
                INSERT INTO results (batch_id, instrument_sn, prediction, t2, q_residual, test_time)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (batch_id, instrument_sn, prediction, T2, Q, test_time))

            pdf.cell(40, 10, fname[:38], 1)
            pdf.cell(30, 10, f"{prediction:.4f}", 1)
            pdf.cell(30, 10, f"{T2:.4f}", 1)
            pdf.cell(30, 10, f"{Q:.4f}", 1)
            pdf.cell(60, 10, test_time, 1)
            pdf.ln()

        conn.commit()
        conn.close()

        report_file = f"{batch_id}.pdf"
        report_path = os.path.join(REPORT_DIR, report_file)
        pdf.output(report_path)

        result = {
            'num_files': len(matched_files),
            'filename': report_file
        }
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(REPORT_DIR, filename), as_attachment=True)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)

