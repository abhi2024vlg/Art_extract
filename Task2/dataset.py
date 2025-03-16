## Import necessary libraries
import os
import csv
import requests
import pandas as pd
from tqdm import tqdm
import time

def download_image(iiif_url, save_path):
    """
    Download an image from an IIIF URL.
    """
    try:
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.1)
        
        # Make the request
        response = requests.get(iiif_url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Check if we got an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            print(f"Error: {iiif_url} returned {content_type} instead of an image")
            return False
        
        # Save the image
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the file was created and has content
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True
        else:
            print(f"Error: Downloaded file {save_path} is empty or missing")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request error downloading {iiif_url}: {e}")
        return False
    except Exception as e:
        print(f"Error downloading {iiif_url}: {e}")
        return False

def main():
    # Set paths and create output directory
    output_dir = "NGA_images"
    metadata_csv = "/Users/abhinavkumar/Desktop/RAG/published_images.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the published_images CSV
    print("Loading image data...")
    images_df = pd.read_csv(metadata_csv)
    
    print(f"Found {len(images_df)} images in the CSV file")
    
    # Print some sample URLs to debug
    print("\nSample iiifURL values:")
    for i, url in enumerate(images_df['iiifthumburl'].head(3)):
        print(f"{i+1}. {url}")
    
    # Create a metadata file for downloaded images
    metadata_output = os.path.join(output_dir, "image_metadata.csv")
    
    with open(metadata_output, 'w', newline='', encoding='utf-8') as meta_file:
        fieldnames = ['filename', 'uuid', 'objectid', 'iiif_url']
        writer = csv.DictWriter(meta_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Download images
        print("\nDownloading images...")
        download_count = 0
        skipped_count = 0
        
        # Take a small sample to test first (remove the .head(20) for full download)
        for _, row in tqdm(images_df.iterrows(), desc="Downloading images"):
            uuid = row['uuid']
            
            # We're going to use the thumbnail URL directly as it's already fully formed
            image_url = row['iiifthumburl']
            object_id = row.get('depictstmsobjectid', 'unknown')
            
            # Create a valid filename
            filename = f"{uuid}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # Download the image
            success = download_image(image_url, save_path)
            
            if success:
                # Write metadata
                writer.writerow({
                    'filename': filename,
                    'uuid': uuid,
                    'objectid': object_id,
                    'iiif_url': image_url
                })
                download_count += 1
            else:
                skipped_count += 1
    
    print(f"\nDownloaded {download_count} images. Skipped {skipped_count} images.")
    print(f"Image metadata saved to {metadata_output}")

if __name__ == "__main__":
    main()
