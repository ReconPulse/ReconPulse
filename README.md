## Hi there 👋 We are Recon Pulse

Below is a polished README file based on your input, formatted for clarity and usability. It assumes the context of a project focused on reconnaissance detection using anomaly-based clustering, incorporating the dataset recommendations and guidance you provided.

---

# Reconnaissance Detection Using Anomaly-Based Clustering

## Overview
This project implements a reconnaissance detection system leveraging anomaly-based clustering techniques. Designed to identify network probing activities (e.g., port scans, network sweeps) in real-time or batch scenarios, the system processes network traffic logs with metadata such as timestamps, IPs, ports, protocols, and packet information. This README provides an overview of recommended datasets, their relevance to the project, and instructions for obtaining and using them. The current date is March 21, 2025, ensuring recommendations reflect up-to-date resources.

## Recommended Datasets
Below are curated datasets suitable for training, testing, and validating the anomaly-based clustering approach. Each dataset includes network traffic with metadata and, where applicable, reconnaissance activities.

### 1. CICIDS-2017
- **Description**: Developed by the Canadian Institute for Cybersecurity (University of New Brunswick), this dataset captures realistic network traffic over five days, including benign behavior and attacks like port scans and probing.
- **Relevance**: 
  - Labeled reconnaissance activities (e.g., port sweeps, SSH probing).
  - Metadata includes timestamps, IPs, ports, protocols, and packet sizes.
  - Features (80+) extracted via Zeek align with preprocessing and clustering needs.
- **Format**: PCAP files and CSV files with extracted features.
- **Access**: [CICIDS-2017 Webpage](https://www.unb.ca/cic/datasets/ids-2017.html) (registration may be required).
- **Use Case**: Ideal for labeled evaluation of clustering accuracy.

### 2. UNSW-NB15
- **Description**: Created by the Australian Centre for Cyber Security, this dataset blends synthetic and real traffic with attacks, including reconnaissance (e.g., network scanning).
- **Relevance**: 
  - Provides raw packets and flow-based features (49 per flow).
  - Includes IPs, ports, packet sizes, and connection durations.
  - Suitable for both batch and streaming analysis.
- **Format**: PCAP files, CSV files with labeled flows, and ground truth data.
- **Access**: [UNSW-NB15 Dataset Page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- **Use Case**: Balances realism and synthetic attack patterns for robust testing.

### 3. HIKARI-2021
- **Description**: A modern dataset from the University of Tokyo, focusing on encrypted traffic with synthetic attacks, including reconnaissance (e.g., probing, brute force).
- **Relevance**: 
  - Features timestamps, IPs, ports, and Zeek-extracted attributes (80+).
  - Captures encrypted reconnaissance, enhancing realism.
  - Compatible with dynamic clustering and stream processing.
- **Format**: PCAP files and CSV files with Zeek features.
- **Access**: See the [MDPI Paper](https://www.mdpi.com/2076-3417/11/21/9945) and its supplementary materials or contact the authors.
- **Use Case**: Perfect for testing against encrypted traffic scenarios.

### 4. MAWI Traffic Archive
- **Description**: Real-world traffic captures from a Japanese backbone network, updated regularly (including 2025 data), provided by the MAWI Working Group.
- **Relevance**: 
  - Unlabeled, raw packet data with timestamps, IPs, ports, and protocols.
  - Reflects authentic network behavior, including potential reconnaissance.
  - Challenges unsupervised clustering and drift detection.
- **Format**: PCAP files.
- **Access**: [MAWI Traffic Archive](http://mawi.wide.ad.jp/mawi/).
- **Use Case**: Simulates real-time streaming and tests adaptability.

## Dataset Selection Guide
- **Labeled Data**: Use **CICIDS-2017** (feature-rich) or **UNSW-NB15** (balanced) for precision/recall evaluation.
- **Encrypted Traffic**: Choose **HIKARI-2021** for modern, encrypted reconnaissance detection.
- **Real-World Streaming**: Select **MAWI Traffic Archive** for unlabeled, real-time analysis.

## How to Obtain and Use
### Obtaining Datasets
1. Visit the provided links for each dataset.
2. Download files (most are free; some require registration or author contact).
3. Verify availability as of March 21, 2025, via hosting institutions.

### Preprocessing
- **Tools**: Use Wireshark, Zeek, or Suricata to process PCAP files into logs if not provided as CSV.
- **Steps**: Clean data, normalize features, and prepare for clustering (e.g., remove noise, standardize timestamps).

### Feature Extraction
- Leverage existing features (e.g., packet counts, flow durations) or extract custom ones:
  - Port scan patterns (e.g., connection rates).
  - DNS query rates or entropy metrics.
- Use Zeek for compatibility across datasets.

### Integration
- **Batch Processing**: Load CSV/PCAP files into your pipeline for offline clustering.
- **Streaming**: Feed data incrementally into Kafka/Spark to test real-time detection.

## Custom Dataset Option
If existing datasets don’t fully meet your needs:
- Generate synthetic traffic with reconnaissance patterns using Scapy or Ostinato.
- Combine with real benign traffic (e.g., MAWI) and process with Zeek/Suricata.

## Project Dependencies
- **Tools**: Wireshark, Zeek, Suricata, Scapy, Ostinato.
- **Streaming**: Kafka, Spark (optional for real-time setups).
- Ensure compatibility with your clustering framework (e.g., Python libraries like scikit-learn).

## Notes
- Datasets are selected for their alignment with network traffic metadata and reconnaissance detection goals.
- Check dataset licenses and terms of use before integration.

## Contact
For questions or contributions, feel free to reach out via the project repository or contact the maintainers.

---

This README is concise, user-friendly, and tailored to a technical audience working on reconnaissance detection. Let me know if you'd like adjustments!
<!--
**ReconPulse/ReconPulse** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
