# ml-notebook-recreation
Recreated machine learning notebook from Units 1–2 using personal notes and API documentation. Includes original notebook, stripped version, and recreated implementation.

# PCA Notebook Notes

> **What is PCA?**
> Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a smaller set of orthogonal components that capture the maximum variance in the dataset.

---

## Cell 1 — Importing Libraries

Imports the required libraries for the notebook.

| Library | Purpose |
| --- | --- |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting graphs |
| `scipy.stats` | Statistical functions |
| `seaborn` style | Improves plot appearance |

---

## Cell 2 — Generating Synthetic Data

Generates a synthetic **2-dimensional dataset**.

- `np.random.seed(1)` ensures reproducibility
- Random points are created and transformed using **matrix multiplication**
- The data is plotted to visualize its distribution

---

## Cell 3 — Applying PCA

Applies **Principal Component Analysis (PCA)**.

- PCA reduces dimensionality by finding directions of **maximum variance**
- `n_components=2` → keep two principal components
- `fit(X)` learns the PCA transformation from the data

```python
pca = PCA(n_components=2)
pca.fit(X)
```

---

## Cell 4 — Displaying PCA Statistics

Displays key PCA statistics after fitting.

| Attribute | Description |
| --- | --- |
| `explained_variance_` | Variance captured by each principal component |
| `components_` | Directions of the principal axes |

```python
pca.explained_variance_
pca.components_
```

---

## Cell 5 — Plotting Principal Component Vectors

Plots the **principal component vectors** on top of the dataset.

- Each vector shows a direction of maximum variance
- **Longer vectors** → more important directions

---

## Cell 6 — PCA with 95% Variance Retention

Uses PCA with automatic component selection to retain **95% of the variance**.

```python
clf = PCA(0.95)
```

> This automatically chooses the number of components needed to keep **95% of the information**.

---

## Cell 7 — Transforming the Dataset

Transforms (compresses) the dataset to a reduced number of dimensions.

```python
X_trans = clf.fit_transform(X)
```

> The result is a **compressed representation** of the original data.

---

## Cell 8 — Reconstructing the Dataset

Reconstructs the dataset back to its original space using:

```python
X_reconstructed = clf.inverse_transform(X_trans)
```

> This converts the compressed data **back to the original space**.

---

## Cell 9 — Loading the Digits Dataset

Loads the **digits dataset** from `sklearn`.

```python
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data    # flattened pixel features (1797 × 64)
y = digits.target  # digit labels (0–9)
```

This dataset contains:

- **1797** images
- **64** pixels per image (8×8 grid)

---

## Cell 10 — Dimensionality Reduction on Digits

Applies PCA to reduce the digits dataset from **64 dimensions to 2 dimensions**.

```python
pca = PCA(n_components=2)
Xproj = pca.fit_transform(X)
```

> This makes the high-dimensional data easier to **visualize**.

---

## Cell 11 — Scatter Plot of PCA Projection

Creates a **scatter plot** of the PCA-reduced digits data.

- Each point = one digit image
- Color = digit label (0–9)

---

## Cell 12 — Displaying a Digit Image

Displays a digit image using `imshow()`.

```python
plt.imshow(digits.images[0], cmap='gray')
```

> Shows how digit images are stored as **pixel matrices**.

---

## Cell 13 — Cumulative Explained Variance

Plots the **cumulative explained variance** as the number of PCA components increases.

Helps determine how many components are needed to retain most information.

> **Example insight:** ~20 components keep about **90% of the variance**

---

## Cell 14 — PCA as Data Compression

Demonstrates PCA as a **data compression** technique.

- Digits are reconstructed using different numbers of PCA components
- **More components** → better image reconstruction quality

---

## Cell 15 — Interactive PCA Visualization

Uses `ipywidgets` to create an **interactive** PCA reconstruction visualization.

- Adjust the number of components with a slider
- See how image quality changes in real time
