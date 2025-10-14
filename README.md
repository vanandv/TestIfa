# IFA Trade Surveillance • Streamlit Prototype

A demo-ready Streamlit app for detecting **Insider/Front‑running Activity (IFA)** using a mix of
rules and an Isolation Forest model. Includes a lightweight “AI agent” that drafts
investigation steps and a compliance memo for each alert.

## 1) Run locally

```bash
# clone and enter the folder
git clone <YOUR_REPO_URL>.git ifa_streamlit_prototype
cd ifa_streamlit_prototype

# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install & run
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed in your terminal.

You can upload a blotter CSV or use the **synthetic data** toggle.

## 2) Deploy on Streamlit Community Cloud

1. Push this folder to **GitHub** (public or private).
2. Go to **share.streamlit.io** (Streamlit Community Cloud) and connect your GitHub.
3. **New app** → pick your repo, `main` (or your branch), and **file path**: `app.py`.
4. Python version: 3.10+ is fine. No secrets are required.
5. Click **Deploy**. The app will build once and stay live; every push to the repo auto‑rebuilds.

> Tip: If your repo is private, grant Streamlit Cloud access and add any needed secrets under
> **App → Settings → Secrets** (not needed for this demo).

## 3) Repo structure

```
ifa_streamlit_prototype/
├─ app.py
├─ requirements.txt
├─ sample_data.csv
├─ .streamlit/
│  └─ config.toml
├─ .gitignore
└─ LICENSE
```

## 4) Sample data

Use `sample_data.csv` to try the app or simply toggle **Use synthetic data**.

**Required columns for uploads:**
- `timestamp`, `symbol`, `side` (BUY/SELL), `price`, `quantity`, `trader_id`, `account_id`, `desk`

**Optional:** `client_order_overlap` (0/1), `pnl_est`

## 5) Notes for compliance demos

- This prototype uses **synthetic features** for client orders/news. Replace with your internal feeds
  to harden the signals.
- The “AI agent” is deterministic (no external LLM). You can swap it for your preferred LLM in a
  guarded way later (e.g., behind an approval toggle).

---

© 2025 MIT License
