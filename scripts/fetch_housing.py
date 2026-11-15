"""Explicit opt-in network download of the real California Housing dataset.
No alternative dataset is substituted when the download fails.
Run: python scripts/fetch_housing.py
"""
from pathlib import Path
from sklearn.datasets import fetch_california_housing
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    data=fetch_california_housing(as_frame=True,download_if_missing=True)
    target=ROOT/'data'/'california_housing.csv'
    data.frame.to_csv(target,index=False)
    print('Downloaded and cached actual California Housing:',data.data.shape)
    print('CSV:',target,'; target unit: 100,000 USD; rows are census block groups.')
