import os
import urllib.request
import sys

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({read_so_far / (1024*1024):.2f} MB of {total_size / (1024*1024):.2f} MB)")
        else:
            sys.stdout.write(f"\rDownloaded {read_so_far / (1024*1024):.2f} MB")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook=report)
    print("\nDownload complete!\n")

def main():
    base_url = "https://huggingface.co/datasets/miminmoons/olist-ecommerce-for-delivery-and-review-prediction/resolve/main/data/"
    files = [
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv"
    ]
    
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    for filename in files:
        url = base_url + filename
        filepath = os.path.join(raw_dir, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"{filename} already exists and is not empty. Skipping download.")
        else:
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    main()
