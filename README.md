# Portfolio Tracker

An app to track investments in stocks, ETFs, Funds, and Crypto.

![Alt text](docs/dashboard.png)


## Description

This app is hosted online on the Streamlit Community Cloud. 

https://portfolio-track.streamlit.app/

## Features

- Stocks, ETFs, Funds and Crypto
- Add trades in the app or manually in a .csv file
- Track portfolio valuation with live market data
- Calculate realized and unrealized P/L
- Automatic FX rates from Frankfurter API
- Calculate P/L in different currencies based on the country of taxation
- Create a bookmark for Streamlit app on iPhone/Android phones


## Build the app locally

Download Python:

```bash
curl -o python-installer.exe https://www.python.org/ftp/python/3.12.1/python-3.12.1-amd64.exe
```


Install Python:

```bash
python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
```


Clone the repo:

```bash
git clone https://github.com/chahatmandviwala-alt/PortfolioTracker.git
```


Install the required modules:

```bash
pip install -r requirements_desktop.txt
```

Run locally in any browser:

```bash
python -m streamlit run main.py
```

Alternatively run locally in a desktop application window:

```bash
pythom main_app.py
```


Note: The local desktop application requires a personal Google OAuth configuration for using the 'Login with Google' option. Use the 'Local login' option or follow the steps below to create your own credentials and update the ./streamlit/secrets.example.toml file.

### Create Google OAuth Credentials

1. Go to Google Cloud Console: https://console.cloud.google.com/

2. Create or select a project.

3. Navigate to: APIs & Services → Credentials

4. Click Create Credentials → OAuth client ID

5. Configure the consent screen if prompted:

6. User type: External

7. Add required app information

8. Add your email as a test user

9. Create an OAuth Client ID:

10. Application type: Web application

11. Authorized redirect URI (example): http://localhost:8501/oauth2callback (Adjust if you modify the port or path.)

12. Save the generated: Client ID & Client Secret

13. Create a random cookie secret and save it:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

14. In your project directory go open: ./streamit/secrets.example.toml

15. Edit the secrets.example.toml file with the information obtianed in steps 11 to 13

16. Save the secrets.example.toml file and rename it to secrets.toml


## Disclosing Vulnerabilities

If you discover an anomaly or calculation mistakes within this application, please send an e-mail to chahat.mandviwala@gmail.com. All vulnerabilities will be promptly addressed.


## Related Projects

- FX Tracker (https://github.com/chahatmandviwala-alt/fx-tracker)
