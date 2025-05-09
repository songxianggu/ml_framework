import kagglehub

# Download latest version
path = kagglehub.dataset_download("ealtman2019/credit-card-transactions")

print("Path to dataset files:", path)