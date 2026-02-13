import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import webview # pip install pywebview

# --- AI LOGIC ---
model = None
def train():
    global model
    df = pd.read_csv('train.csv', nrows=10000)
    features = ['Age', 'Sex', 'Chest pain type', 'Cholesterol', 'BP', 'Max HR']
    X = df[features]
    y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    model = RandomForestClassifier(n_estimators=20, max_depth=7, random_state=42)
    model.fit(X, y)

class Api:
    def predict(self, data):
        # Data comes from HTML form
        df_input = pd.DataFrame([data])
        risk = model.predict_proba(df_input)[0][1] * 100
        return round(risk, 2)

# --- HTML/CSS GUI ---
html_ui = """
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f4f7f6; font-family: 'Segoe UI', sans-serif; }
        .container { max-width: 500px; margin-top: 30px; }
        .card { border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: none; }
        .header { background: #c62828; color: white; border-radius: 15px 15px 0 0; padding: 20px; text-align: center; }
        .form-label { font-weight: 600; color: #444; font-size: 0.9rem; }
        .btn-predict { background: #c62828; color: white; width: 100%; padding: 12px; border-radius: 8px; border: none; font-weight: bold; margin-top: 20px; }
        #result { display: none; text-align: center; margin-top: 20px; padding: 15px; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header"><h3>❤️ HeartGuard AI</h3></div>
            <div class="p-4">
                <div class="mb-3">
                    <label class="form-label">Age</label>
                    <input type="number" id="Age" class="form-control" value="45">
                </div>
                <div class="row">
                    <div class="col"><label class="form-label">Sex (1=M, 0=F)</label><input type="number" id="Sex" class="form-control" value="1"></div>
                    <div class="col"><label class="form-label">Chest Pain (1-4)</label><input type="number" id="CP" class="form-control" value="1"></div>
                </div>
                <div class="mt-3"><label class="form-label">Cholesterol</label><input type="number" id="Chol" class="form-control" value="200"></div>
                <div class="mt-3"><label class="form-label">Blood Pressure</label><input type="number" id="BP" class="form-control" value="120"></div>
                <div class="mt-3"><label class="form-label">Max Heart Rate</label><input type="number" id="HR" class="form-control" value="150"></div>
                
                <button class="btn-predict" onclick="runAI()">ANALYZE DATA</button>
                
                <div id="result">
                    <h2 id="score">0%</h2>
                    <p id="status"></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runAI() {
            const data = {
                'Age': parseFloat(document.getElementById('Age').value),
                'Sex': parseFloat(document.getElementById('Sex').value),
                'Chest pain type': parseFloat(document.getElementById('CP').value),
                'Cholesterol': parseFloat(document.getElementById('Chol').value),
                'BP': parseFloat(document.getElementById('BP').value),
                'Max HR': parseFloat(document.getElementById('HR').value)
            };
            const risk = await pywebview.api.predict(data);
            document.getElementById('result').style.display = 'block';
            document.getElementById('score').innerText = risk + '% Risk';
            
            const resBox = document.getElementById('result');
            if(risk > 70) { resBox.className = 'alert alert-danger'; }
            else if(risk > 30) { resBox.className = 'alert alert-warning'; }
            else { resBox.className = 'alert alert-success'; }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    train() # Phase 1
    api = Api()
    # Phase 3: Deployment
    window = webview.create_window(
    "HeartGuard AI Professional",   # 👈 App name at top-left
    html=html_ui,
    js_api=api,
    width=1200,
    height=750,
    resizable=True,      # Enables maximize button
    fullscreen=False,    # MUST be False
    frameless=False      # Shows title bar with Min/Max/Close
)
    webview.start()
