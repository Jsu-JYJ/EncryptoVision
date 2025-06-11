import pandas as pd
import numpy as np
from PIL import Image
import os
import ast
import hashlib
import ipaddress


# Convert IP address to a fixed-length numeric representation
def ip_to_bytes(ip_str, size=16):
    try:
        ip = ipaddress.ip_address(ip_str)
        if isinstance(ip, ipaddress.IPv4Address):
            return list(ip.packed) + [0] * (size - 4)  # Pad IPv4 to 16 bytes
        elif isinstance(ip, ipaddress.IPv6Address):
            return list(ip.packed)  # IPv6 is already 16 bytes
    except ValueError:
        print(f"Invalid IP address: {ip_str}")
        return [0] * size  # Return zeros if IP is invalid


# Normalize and resize/pad data
def normalize_and_resize(data, size=256):
    # If data is a string representation of a list or matrix, parse it into a Python object
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)  # Safely parse string to list or matrix
        except (ValueError, SyntaxError):
            print(f"Error parsing field: {data}")
            return [0] * size

    # Flatten multi-dimensional data into a 1D list
    if isinstance(data, list) and all(isinstance(item, list) for item in data):  # Check if it's a 2D list
        data = [item for sublist in data for item in sublist]

    # Convert to list of numeric values
    if isinstance(data, list):
        data = [float(x) for x in data]
    elif isinstance(data, (int, float)):
        data = [float(data)]  # Convert single value to a list
    else:
        data = []

    # Normalize to [0, 255]
    if len(data) > 0:
        min_val = min(data)
        max_val = max(data)
        if max_val - min_val != 0:
            data = [(x - min_val) / (max_val - min_val) * 255 for x in data]
        else:
            data = [0] * len(data)  # Normalize to zero if all values are the same

    # Trim or pad to desired size
    if len(data) > size:
        data = data[:size]
    else:
        data += [0] * (size - len(data))

    return np.array(data, dtype=np.uint8)


# Generate RGB image
def generate_rgb_image(row, index):
    try:
        # Extract fields and parse them
        src_ip = row['SRC_IP']
        dst_ip = row['DST_IP']

        # Check if SRC_IP and DST_IP are valid strings
        if not isinstance(src_ip, str) or not isinstance(dst_ip, str):
            print(f"Invalid SRC_IP or DST_IP for row {index}")
            return None

        phist_src_sizes = ast.literal_eval(row['PHIST_SRC_SIZES'])
        phist_dst_sizes = ast.literal_eval(row['PHIST_DST_SIZES'])
        phist_src_ipt = ast.literal_eval(row['PHIST_SRC_IPT'])
        phist_dst_ipt = ast.literal_eval(row['PHIST_DST_IPT'])
        ppi = ast.literal_eval(row['PPI'])

        # Check if PPI is a 2D list
        if not isinstance(ppi, list) or not all(isinstance(sublist, list) for sublist in ppi):
            print(f"Invalid PPI format for row {index}")
            return None

        # R Channel: SRC_IP, DST_IP, PHIST_SRC_SIZES, PHIST_DST_SIZES
        r_channel_data = []
        r_channel_data.extend(ip_to_bytes(src_ip))  # Convert SRC_IP to bytes
        r_channel_data.extend(ip_to_bytes(dst_ip))  # Convert DST_IP to bytes
        r_channel_data.extend(normalize_and_resize([row['DST_ASN']]))
        r_channel_data.extend(normalize_and_resize([row['SRC_PORT']]))
        r_channel_data.extend(normalize_and_resize([row['DST_PORT']]))
        r_channel_data.extend(normalize_and_resize([row['PROTOCOL']]))
        r_channel_data.extend(normalize_and_resize(phist_src_sizes))
        r_channel_data.extend(normalize_and_resize(phist_dst_sizes))
        r_channel = normalize_and_resize(r_channel_data).reshape((16, 16))

        # G Channel: PHIST_SRC_IPT, PHIST_DST_IPT, PPI
        g_channel_data = []
        g_channel_data.extend(normalize_and_resize(phist_src_ipt))
        g_channel_data.extend(normalize_and_resize(phist_dst_ipt))
        g_channel_data.extend(normalize_and_resize(ppi))  # Process 2D matrix form of PPI
        g_channel = normalize_and_resize(g_channel_data).reshape((16, 16))

        # B Channel: Other fields
        b_channel_data = []
        b_channel_data.extend(normalize_and_resize([row['QUIC_VERSION']]))
        b_channel_data.extend(normalize_and_resize([row['BYTES']]))
        b_channel_data.extend(normalize_and_resize([row['BYTES_REV']]))
        b_channel_data.extend(normalize_and_resize([row['PACKETS']]))
        b_channel_data.extend(normalize_and_resize([row['PACKETS_REV']]))
        b_channel_data.extend(normalize_and_resize([row['DURATION']]))
        b_channel_data.extend(normalize_and_resize([row['PPI_LEN']]))
        b_channel_data.extend(normalize_and_resize([row['PPI_DURATION']]))
        b_channel_data.extend(normalize_and_resize([row['PPI_ROUNDTRIPS']]))
        b_channel_data.extend(normalize_and_resize([int(row['FLOW_ENDREASON_IDLE'])]))
        b_channel_data.extend(normalize_and_resize([int(row['FLOW_ENDREASON_ACTIVE'])]))
        b_channel_data.extend(normalize_and_resize([int(row['FLOW_ENDREASON_OTHER'])]))

        # Hash QUIC_SNI and normalize
        quic_sni_hash = int(hashlib.md5(row['QUIC_SNI'].encode()).hexdigest(), 16) % 256
        b_channel_data.extend([quic_sni_hash])

        b_channel = normalize_and_resize(b_channel_data).reshape((16, 16))

        # Combine into an RGB image
        image = np.dstack([r_channel, g_channel, b_channel]).astype(np.uint8)
        return image
    except Exception as e:
        print(f"Error generating image for row {index}: {e}")
        return None


# Create folders based on CATEGORY and save the image as PNG
def save_image_as_png(image, category, app, src_ip, file_name_suffix, output_dir="output"):
    try:
        # Create CATEGORY folder
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # Construct file name
        file_name = f"{category}-{app}-{src_ip}-{file_name_suffix}.png"
        file_path = os.path.join(category_dir, file_name)

        # Save using Pillow as PNG
        img = Image.fromarray(image, 'RGB')
        img.save(file_path)

        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Failed to save {file_path}: {e}")


def main():
    csv_file = 'test.csv'  # Replace with your CSV file path
    chunk_size = 10000  # Number of rows per chunk

    try:
        # Load data in chunks using chunksize
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            # Ensure required columns exist
            required_columns = {'SRC_IP', 'DST_IP', 'DST_ASN', 'SRC_PORT', 'DST_PORT', 'PROTOCOL',
                                'PHIST_SRC_SIZES', 'PHIST_DST_SIZES', 'QUIC_VERSION',
                                'PHIST_SRC_IPT', 'PHIST_DST_IPT', 'PPI', 'BYTES', 'BYTES_REV',
                                'PACKETS', 'PACKETS_REV', 'DURATION', 'PPI_LEN', 'PPI_DURATION',
                                'FLOW_ENDREASON_IDLE', 'FLOW_ENDREASON_ACTIVE', 'FLOW_ENDREASON_OTHER',
                                'QUIC_SNI', 'CATEGORY', 'APP', 'PPI_ROUNDTRIPS'}
            if not required_columns.issubset(chunk.columns):
                print(f"Missing required columns: {required_columns - set(chunk.columns)}")
                continue

            # Filter criteria: BYTES_REV > 4096 and DURATION < 3
            filtered_chunk = chunk[(chunk['BYTES_REV'] > 4096) & (chunk['DURATION'] < 3)]

            # Process each row and generate images
            for index, row in filtered_chunk.iterrows():
                image = generate_rgb_image(row, index)

                if image is None:
                    print(f"Failed to generate image for row {index}")
                    continue

                category = row['CATEGORY']
                app = row['APP']
                src_ip = row['SRC_IP']

                # Add unique identifier to avoid filename conflicts
                file_name_suffix = f"{index % 10000:04d}"  # Use row index as suffix
                save_image_as_png(image, category, app, src_ip, file_name_suffix)

    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()