# Industrial Anomaly Pattern Discovery
An unsupervised machine learning project for discovering **recurring patterns in industrial sensor anomalies**.
Instead of treating every anomaly as an isolated event, the system groups anomalous time windows according to their sensor behavior. This helps identify repeated patterns that can later be investigated by domain experts.
> **Important:** The discovered clusters represent patterns in sensor behavior. They are not directly interpreted as machine failure types, root causes, or causal relationships.

## Problem Statement
Industrial systems continuously generate large amounts of multivariate sensor data. Traditional anomaly detection can identify unusual observations, but detecting an anomaly does not necessarily explain whether similar anomalous behavior has occurred before.
This project investigates whether **unsupervised machine learning can discover recurring patterns among industrial anomalies**.
The main question is:
> **Can anomalous periods be grouped into meaningful, recurring sensor-behavior patterns without predefined failure labels?**

## Dataset
The project uses the **TiNA (Time-series Industrial Anomaly) Dataset**, an industrial time-series dataset containing anonymized sensor measurements and alarm information.
The complete dataset contains more than 38 million observations across 108 features and covers a long operating period.
The full dataset is **not included in this repository** because of its large size.

### Current Feasibility Experiment
The initial experiment uses a **7-day subset**:
* **Period:** September 6–12, 2017
* **Observations:** 592,426
* **Sensor features:** 105
* **Time windows:** 9,869
* **Anomalous windows:** 494
* **Anomaly proportion:** approximately 5%

The 7-day subset was selected as an initial feasibility experiment before scaling the analysis to a larger portion of the dataset.
## Machine Learning Pipeline
```text
Raw Industrial Sensor Data
          ↓
Chronological Sorting
          ↓
Time-Based Windowing
          ↓
Window-Level Feature Extraction
          ↓
Standardization
          ↓
Isolation Forest
          ↓
Anomalous Windows
          ↓
PCA
          ↓
K-Means Clustering
          ↓
Recurring Anomaly Patterns
          ↓
Post-Hoc Alarm Analysis
```

### 1. Time-Based Windowing
The raw sensor observations are converted into **5-minute windows with a 1-minute stride**.
Actual timestamps are used for windowing rather than assuming that every row represents a fixed time interval.

### 2. Feature Extraction
For each sensor within each time window, statistical features are calculated:
* Mean
* Standard deviation
* Minimum
* Maximum
* Range
* Change from the beginning to the end of the window
With 105 sensor features and 6 statistics per sensor, this produces **630 window-level features**.

### 3. Anomaly Detection
An **Isolation Forest** is used to identify unusual time windows.
The experiment uses:
* `n_estimators = 200`
* `contamination = 0.05`
* `random_state = 42`
The model identifies anomalous windows without requiring predefined anomaly labels.

### 4. Dimensionality Reduction
PCA is applied to the 630-dimensional window representation.
**13 principal components** explain approximately **90.83% of the variance**.
A separate two-dimensional PCA representation is used for visualization.

### 5. Clustering
Only the anomalous windows are passed to K-Means clustering.
K values from 2 to 5 were evaluated.
| K | Silhouette Score | Davies-Bouldin Score |
| - | ---------------: | -------------------: |
| 2 |           0.6918 |               0.4922 |
| 3 |       **0.7118** |               0.5664 |
| 4 |           0.6837 |               0.6021 |
| 5 |           0.6540 |               0.7637 |

The experiment uses **K = 3** for exploratory pattern analysis.

## Discovered Patterns
The K-Means analysis produced three clusters of anomalous sensor behavior.
| Cluster   | Windows | % of Anomalies | Alarm Association |
| --------- | ------: | -------------: | ----------------: |
| Cluster 0 |      75 |         15.18% |            20.00% |
| Cluster 1 |     252 |         51.01% |            14.29% |
| Cluster 2 |     167 |         33.81% |            45.51% |

The clusters show different statistical behavior across several sensors.
The most discriminative sensors identified during the analysis include:
* `FEATURE70`
* `FEATURE68`
* `FEATURE69`
* `FEATURE43`
* `FEATURE71`
* `FEATURE47`
* `FEATURE51`
* `FEATURE52`

For example, `FEATURE70`, `FEATURE68`, and `FEATURE69` show substantial differences in measures such as minimum, maximum, range, and standard deviation across the discovered clusters.
These differences indicate that the clusters are capturing **different multivariate sensor-behavior patterns**.

## Cluster Stability
To check whether the clustering result depended heavily on the initial K-Means seed, the clustering experiment was repeated using multiple random seeds.
The mean pairwise Adjusted Rand Index across the tested seeds was:
**ARI = 1.00**
This indicates that the clustering assignments were highly consistent across the tested K-Means initializations.

## Alarm Analysis
Alarm information is used **after** anomaly detection and clustering.
It is not used as an input feature for the unsupervised ML models.
This allows the experiment to ask:
> "Do the discovered sensor-behavior patterns show different relationships with recorded alarms?"
For example, Cluster 2 had an alarm association of approximately **45.51%**, compared with 20.00% for Cluster 0 and 14.29% for Cluster 1.
This is an observed association only. It does **not** establish that a particular cluster causes an alarm or represents a particular physical failure.

## Current Feasibility Result
The initial 7-day experiment provides evidence that the proposed approach is **promising for further investigation**.
The main supporting observations are:
* The anomaly detector identified a manageable set of anomalous windows.
* K-Means produced reasonably separated clusters.
* The K=3 solution achieved a silhouette score of **0.7118**.
* Cluster assignments were highly stable across tested random seeds (**ARI = 1.00**).
* Several sensor features showed strong differences between clusters.
* The clusters exhibited different levels of post-hoc alarm association.

These results support continuing the project with a larger dataset and an interactive investigation interface.
 
## Project Structure

```text
CodeSprint-Week-2/
│
├── tina_experiment.py
│
├── outputs/
│   └── tina_experiment/
│       ├── analysis_report.txt
│       ├── cluster_interpretation_report.txt
│       ├── cluster_profiles.csv
│       ├── cluster_0_representatives.csv
│       ├── cluster_1_representatives.csv
│       ├── cluster_2_representatives.csv
│       ├── top_cluster_features.csv
│       ├── top_cluster_features_corrected.csv
│       ├── top_discriminative_sensors.csv
│       └── visualization files
│
└── README.md
```

The raw TiNA dataset is intentionally excluded from the repository.

---

## Technologies Used

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Isolation Forest
* PCA
* K-Means
* StandardScaler

### Data Processing

* PyArrow
* Parquet

### Visualization

* Matplotlib

---

## Running the Experiment

### 1. Clone the repository

```bash
git clone https://github.com/Dikshitha2607/CodeSprint-Week-2.git
cd CodeSprint-Week-2
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow
```

### 4. Provide the TiNA dataset

Download the TiNA dataset separately and place the required daily Parquet files in the expected dataset directory.

The raw dataset should not be committed to GitHub.

### 5. Run the experiment

```bash
python tina_experiment.py
```

The generated analysis files and visualizations are stored under:

```text
outputs/tina_experiment/
```

---

## Limitations

The current results should be interpreted as a **feasibility experiment**, not as a final industrial diagnostic system.

### Limited time period

The current experiment uses only seven days of data. A larger time period is required to determine whether the discovered patterns remain stable across different operating conditions and time periods.

### Anonymous sensor features

The sensor columns are anonymized. Therefore, the project can identify statistical behavior differences but cannot assign physical meanings to features such as `FEATURE70` or `FEATURE43` without additional domain information.

### No confirmed failure labels

The clustering is unsupervised. The discovered clusters have not been validated as specific machine failure categories.

### No causal interpretation

An association between a cluster and an alarm does not prove that the cluster caused the alarm.

### Anomaly detector dependence

The clustering analysis operates on the anomalous windows identified by the Isolation Forest. Different anomaly-detection configurations could produce a different set of windows.

---

## Future Work

The next stages of the project can include:

1. Expand the analysis from the initial 7-day experiment to a larger TiNA time period.
2. Test whether discovered patterns remain stable across different periods.
3. Build an interactive anomaly timeline.
4. Provide cluster-level sensor comparisons.
5. Allow investigation of representative anomalous windows.
6. Compare discovered patterns with available alarm and maintenance information as post-hoc context.
7. Provide domain experts with interpretable sensor-level evidence for further investigation.

---

## Key Takeaway

The goal of this project is not simply to answer:

> **"Is this observation anomalous?"**

Instead, it investigates:

> **"When anomalies occur, do they form recurring patterns of multivariate sensor behavior?"**

The initial TiNA experiment suggests that unsupervised machine learning can identify stable and distinguishable groups within anomalous time windows, providing a basis for further investigation at a larger scale.

## Dataset Reference

TiNA — Time-series Industrial Anomaly Dataset

Official dataset information:

https://dasci.es/opendata/tina-time-series-industrial-anomaly-dataset/

Official repository:

https://github.com/ari-dasci/OD-TINA
