import os
import shutil
import subprocess
from tqdm import tqdm
import scapy.all as scapy

clean_protocols = 'not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp'
exact_3 = False


def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


def process_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            class_path = os.path.join(root, dir_name)
            print(f"\n----------\nProcessing Class: [{dir_name}]\n----------")
            process_class(class_path)


def process_class(class_path):
    for root, _, files in os.walk(class_path):
        for file_name in tqdm(files, desc="Processing PCAP files"):
            full_path = os.path.join(root, file_name)

            # Step 1: rename .pcapng -> .pcap
            if file_name.endswith('.pcapng'):
                new_path = full_path.replace('.pcapng', '.pcap')
                shutil.move(full_path, new_path)
                full_path = new_path

            base_name = os.path.splitext(full_path)[0]

            # Step 2: truncate large pcap
            if exact_3:
                truncated_path = f"{base_name}_60000.pcap"
                subprocess.run(["tcpdump", "-r", full_path, "-w", truncated_path, "-c", "60000"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                safe_remove(full_path)
                full_path = truncated_path

            # Step 3: clean unwanted protocols
            cleaned_path = f"{base_name}.clean.pcap"
            subprocess.run(["E:\\Apps\\Wireshark\\tshark.exe", "-F", "pcap", "-r", full_path, "-Y", clean_protocols, "-w", cleaned_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # safe_remove(full_path)
            shutil.move(cleaned_path, full_path)

            # Step 4: split by session using SplitCap
            session_dir = os.path.dirname(full_path)
            subprocess.run(["E:\\paper\\preprocess\\tools\\SplitCap.exe", "-r", full_path, "-s", "session", "-o", session_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            safe_remove(full_path)

            # Step 5: process each session pcap
            session_files = [f for f in os.listdir(session_dir) if f.endswith('.pcap')]
            for session_file in session_files:
                session_full_path = os.path.join(session_dir, session_file)
                process_session(session_full_path, session_dir)


def process_session(session_path, session_dir):
    try:
        packets = scapy.rdpcap(session_path)
    except Exception as e:
        print(f"Error reading {session_path}: {e}")
        safe_remove(session_path)
        return

    if exact_3:
        # 拆分成多个 flow，每个包含 3 个包
        total_packets = len(packets)
        num_flows = total_packets // 3
        if num_flows < 1:
            safe_remove(session_path)
            return

        for i in range(num_flows):
            flow_packets = packets[i*3 : (i+1)*3]
            flow_path = session_path.replace(".pcap", f"_flow{i}.pcap")
            scapy.wrpcap(flow_path, flow_packets)

        safe_remove(session_path)
    else:
        # 保留至少 3 个包的流
        if len(packets) < 1:
            safe_remove(session_path)


if __name__ == "__main__":
    pcap_path = ""
    print("Starting flow processing pipeline...")
    process_directory(pcap_path)
    print("Flow processing completed.")