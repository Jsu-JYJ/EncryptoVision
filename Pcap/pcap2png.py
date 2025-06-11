import os
import glob
import binascii
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import numpy
import random


def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_3hp_list(pcap_dir):
    """
    Reads a .pcap file and extracts the first 3 packets.
    For each packet:
    - Extracts the IP header (and lower layer headers) as hex
    - Extracts the payload (Raw data) as hex
    - Truncates or pads both header and payload to fixed lengths
    Returns concatenated hex string from all 3 packets
    """
    packets = scapy.rdpcap(pcap_dir)
    data = []

    for packet in packets:
        # Extract header (IP layer and lower protocol headers)
        try:
            header_bytes = bytes(packet['IP'])
        except:
            continue  # Skip this packet if no IP layer

        header_hex = binascii.hexlify(header_bytes).decode()

        # Extract payload (application layer Raw data)
        try:
            payload_bytes = bytes(packet['Raw'])
            payload_hex = binascii.hexlify(payload_bytes).decode()
            # Remove payload part from header if present
            header_hex = header_hex.replace(payload_hex, '')
        except:
            payload_hex = ''

        # Truncate or pad header to 80 bytes (160 hex chars)
        if len(header_hex) > 160:
            header_hex = header_hex[:160]
        else:
            header_hex += '0' * (160 - len(header_hex))

        # Truncate or pad payload to 176 bytes (352 hex chars)
        if len(payload_hex) > 352:
            payload_hex = payload_hex[:352]
        else:
            payload_hex += '0' * (352 - len(payload_hex))

        data.append((header_hex, payload_hex))

        # Keep at most 3 packets
        if len(data) >= 3:
            break

    # Pad with empty packets if less than 3
    while len(data) < 3:
        data.append(('0' * 160, '0' * 352))

    # Concatenate all headers and payloads
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p

    return final_data  # Returns a string of 1536 hex chars → 768 bytes


def generate_rgb_image_from_hex(hex_str):
    """
    Input: Hex string of length 1536 (3×512 chars)
    Output: PIL Image (16x16 RGB)
    """
    if len(hex_str) != 1536:
        raise ValueError(f"Expected 1536 hex chars, got {len(hex_str)}")

    # Convert every two characters to one byte
    byte_values = [int(hex_str[i:i + 2], 16) for i in range(0, len(hex_str), 2)]

    # Reshape into 3 channels, each 256 bytes → 16x16 image
    r = numpy.array(byte_values[0:256]).reshape((16, 16))
    g = numpy.array(byte_values[256:512]).reshape((16, 16))
    b = numpy.array(byte_values[512:768]).reshape((16, 16))

    # Stack into RGB image
    rgb_array = numpy.stack([r, g, b], axis=2).astype('uint8')
    image = Image.fromarray(rgb_array)

    return image


def RGB_generator(
        flows_pcap_path,
        output_path,
        max_samples_per_class=1000,
        train_ratio=0.9,
        seed=42
):
    """
    Generates RGB images from .pcap files and splits them into train/test sets.

    Parameters:
    - flows_pcap_path: Root directory containing subfolders of .pcap files
    - output_path: Directory where train/test folders will be created
    - max_samples_per_class: Maximum number of samples per class to process
    - train_ratio: Ratio of training set size over total samples
    - seed: Random seed for shuffling
    """

    # Set random seed
    random.seed(seed)

    # Get all .pcap files under immediate subdirectories (only second-level directories)
    class_to_files = {}
    for root, dirs, files in os.walk(flows_pcap_path):
        # Process only first-level subdirectories (i.e., second-level directories)
        if root == flows_pcap_path:
            for d in dirs:
                class_name = d
                dir_path = os.path.join(root, d)
                pcap_files = glob.glob(os.path.join(dir_path, "*.pcap"))
                if pcap_files:
                    class_to_files[class_name] = pcap_files
            break  # Do not go deeper into nested directories

    print(f"Found {len(class_to_files)} classes.")

    # Create output directories
    train_dir = os.path.join(output_path, "train")
    test_dir = os.path.join(output_path, "test")
    makedir(train_dir)
    makedir(test_dir)

    # Process each class
    for class_name, files in class_to_files.items():
        total_files = len(files)
        selected_files = files[:max_samples_per_class] if total_files > max_samples_per_class else files
        print(f"Class '{class_name}': Using {len(selected_files)} out of {total_files} files.")

        random.shuffle(selected_files)  # Shuffle order
        split_idx = int(len(selected_files) * train_ratio)
        train_files = selected_files[:split_idx]
        test_files = selected_files[split_idx:]

        # Create class-specific train/test subdirectories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        makedir(train_class_dir)
        makedir(test_class_dir)

        # Process train files
        for f in tqdm(train_files, desc=f"Processing train/{class_name}"):
            content = read_3hp_list(f)
            if not content:
                continue
            try:
                img = generate_rgb_image_from_hex(content)
                out_path = os.path.join(train_class_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
                img.save(out_path)
            except Exception as e:
                print(f"Error processing {f}: {e}")

        # Process test files
        for f in tqdm(test_files, desc=f"Processing test/{class_name}"):
            content = read_3hp_list(f)
            if not content:
                continue
            try:
                img = generate_rgb_image_from_hex(content)
                out_path = os.path.join(test_class_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
                img.save(out_path)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    print("✅ Train/Test split completed.")


if __name__ == "__main__":
    pcap_path = ""
    output_path = ""
    RGB_generator(pcap_path, output_path, 6000, train_ratio=0.9, seed=42)