# Deploy the Customer Clustering Dashboard on Streamlit Community Cloud

This project now has two ways to run:

- **Streamlit website:** `streamlit_app.py` (interactive dashboard)
- **Original command-line program:** `main.py` (terminal output and saved charts)

## 1. Test the Streamlit app on your computer

Open the `Clustering_Project` folder in VS Code. Open Terminal and run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

If your current virtual environment is broken, remove the `.venv` folder and create a fresh one. To use the default Python registered with the Windows Python Launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To explicitly use Python 3.14, replace the first command with `py -3.14 -m venv .venv`. Streamlit Community Cloud currently supports released Python versions that still receive security updates; use the same version locally and in the cloud where possible.

Streamlit should open a browser tab. If it does not, open the Local URL printed in the terminal, usually `http://localhost:8501`.

The first run needs internet access to retrieve the official UCI dataset. The app caches the dataset. You can also upload a CSV containing these six column names:

- `Fresh`
- `Milk`
- `Grocery`
- `Frozen`
- `Detergents_Paper`
- `Delicassen`

## 2. Upload the project to GitHub

1. Sign in to GitHub and create a **new repository** named `customer-clustering-streamlit`.
2. Choose Public if you want the easiest Community Cloud setup. A private repository can also be used after granting Community Cloud access.
3. Do not initialize the repository with an additional README if you plan to push this existing project folder from VS Code.
4. In VS Code, open the `Clustering_Project` folder.
5. Open Terminal and run the following commands, replacing `YOUR_GITHUB_USERNAME` with your GitHub username:

```powershell
git init
git add .
git commit -m "Add Streamlit customer clustering dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/customer-clustering-streamlit.git
git push -u origin main
```

If Git asks you to sign in, complete the browser sign-in flow. If `git` is not recognized, install Git for Windows and restart VS Code: https://git-scm.com/download/win

## 3. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/.
2. Sign in with GitHub.
3. Authorize Streamlit to access the repository.
4. Click **Create app**.
5. Select your repository: `YOUR_GITHUB_USERNAME/customer-clustering-streamlit`.
6. Select branch: `main`.
7. Set **Main file path** to `streamlit_app.py`.
8. In **Advanced settings**, choose the same Python version used for local testing. Python 3.14 is supported by current Streamlit tooling when listed in the version selector.
9. Click **Deploy**.

Streamlit Community Cloud will install packages from `requirements.txt` and build the app. When deployment finishes, it will provide a public `*.streamlit.app` URL.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## 4. Updating the deployed app

After editing files in VS Code, push the changes:

```powershell
git add .
git commit -m "Update clustering dashboard"
git push
```

Community Cloud watches the connected GitHub repository and automatically updates the app after pushed changes.

## Notes

- The iOS-inspired Liquid Glass dashboard includes a manual K selector (K=2 through the selected upper limit), automatic silhouette recommendation, and interactive DBSCAN Epsilon / Minimum Samples controls.
- The UCI `Channel` column has two categories and `Region` has three; these metadata categories do not limit the number of K-Means clusters. Clustering uses the six spending columns.
- The K-distance graph displays the current Epsilon reference line.
- The clustering runs on all six spending features. PCA is used only to create two-dimensional visualization charts.
- The uploaded/downloaded results are CSV files.
- The app is public to anyone who has its URL unless you configure supported access controls. Do not upload private customer data unless you have permission to share it.


## Updating the deployed dashboard

After editing the project locally, verify it with `python -m streamlit run streamlit_app.py`, then commit and push the updated `streamlit_app.py` and project files to the same GitHub branch connected to Streamlit Community Cloud. Streamlit will detect the new commit and rebuild the app. The dashboard uses a fixed midnight Liquid Glass appearance with visible cyan/violet neon accents, an emphasized sidebar control, subtle animated ambient effects, and responsive Plotly charts for mobile screens. There is no Light/Dark switch.
