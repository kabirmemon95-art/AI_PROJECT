# Customer Intelligence — K-Means + DBSCAN

An interactive, glass-style Streamlit dashboard for exploring the UCI Wholesale Customers dataset. The project keeps the original command-line implementation in `main.py` and adds a browser-based dashboard in `streamlit_app.py`.

## Dashboard features

- Midnight Liquid Glass interface with translucent navy surfaces, crisp cyan/violet neon accents, stronger frosted borders, specular highlights, soft depth shadows, and subtle ambient motion. The appearance is intentionally fixed; there is no Light/Dark switch.
- Prominent native sidebar collapse/expand control with a larger frosted-glass treatment.
- Mobile-responsive charts: Plotly resizes to the available viewport, chart columns stack on narrow screens, and tabs remain horizontally swipeable.
- Official UCI dataset loader and CSV upload option.
- K-Means manual K selection (2–10 by default; increase the tested maximum up to 12) or automatic silhouette-based recommendation.
- Elbow and silhouette evaluation charts.
- DBSCAN Epsilon and Minimum Samples controls.
- K-distance curve with a live Epsilon reference line.
- PCA cluster maps, customer counts, cluster spending profiles, and CSV downloads.
- The original terminal program `main.py` is preserved.

### Appearance and mobile layout

The dashboard uses one consistent midnight-glass appearance with higher-contrast text, cyan and violet neon details, visibly frosted panels, and a luminous sidebar collapse/expand control. On phone-sized screens, paired chart panels stack into a single column, Plotly charts use responsive sizing, KPI cards form a compact grid, and the dashboard tabs can be swiped horizontally.

## Important: UCI categories are not the K setting

The UCI Wholesale Customers dataset has 440 records. It contains six annual-spending columns used as clustering inputs. It also includes `Channel` (Horeca / Retail) and `Region` metadata. `Channel` has two categories and `Region` has three categories, but neither one restricts K-Means. The K setting is chosen by the analyst and can be tested at 2, 3, 4, 5, and so on. This dashboard uses only the six spending columns for the clustering calculations.

UCI source: https://archive.ics.uci.edu/dataset/292/wholesale+customers

## Run locally in VS Code

### Option A — automatic Windows setup

1. Extract the ZIP.
2. Open the `Clustering_Project` folder in VS Code.
3. Double-click `setup_streamlit.bat`, or run it in the VS Code terminal:

```powershell
.\setup_streamlit.bat
```

The setup script uses the default Python registered with the Windows Python Launcher, creates `.venv`, installs dependencies, and starts the dashboard.

### Option B — manual setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To create the environment with Python 3.14 explicitly, use `py -3.14 -m venv .venv` instead of `py -m venv .venv`. Use the same Python version locally and in Streamlit Community Cloud when possible.

Open the local URL printed in the terminal, usually http://localhost:8501. The first UCI dataset load requires internet access. You can instead upload a CSV that contains all six spending columns:

- `Fresh`
- `Milk`
- `Grocery`
- `Frozen`
- `Detergents_Paper`
- `Delicassen`

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app** and choose your repository and `main` branch.
4. Set the main file path to `streamlit_app.py`.
5. In Advanced settings, select the same Python version used for local testing (Python 3.14 is supported by current Streamlit tooling when available in the dropdown).
6. Deploy.

See `STREAMLIT_DEPLOYMENT_GUIDE.md` for the full GitHub and deployment commands. Official documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Original command-line implementation

The original CLI program is still available:

```powershell
python main.py
```

Or run `setup_and_run.bat` for the first CLI setup and `run_project.bat` for later runs.

## Method

1. Select the six numeric spending categories.
2. Apply `log1p` to reduce right skew.
3. Standardize the six features.
4. Evaluate K-Means for K values from 2 to the selected maximum.
5. Fit K-Means using the manually selected K or the best silhouette recommendation.
6. Fit DBSCAN using the selected Epsilon and Minimum Samples values.
7. Use PCA only to visualize the six-dimensional data in two dimensions.
8. Compare cluster counts, noise points, silhouette scores, and average spending profiles.

## Interpretation notes

- K-Means assigns each record to one of K clusters.
- DBSCAN can label outlier/noise records as `-1`.
- DBSCAN silhouette is calculated only on non-noise records when at least two non-noise clusters exist; compare it cautiously with K-Means.
- PCA is for visualization only. Both clustering algorithms use all six scaled spending features.
