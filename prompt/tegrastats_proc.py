#!/usr/bin/env python3
import argparse
import re
import csv
from datetime import datetime

def parse_line(line):
    data = {}
    
    # Extract timestamp if present
    timestamp_match = re.match(r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
        data['timestamp'] = timestamp_match.group(1)
        line = line[len(timestamp_match.group(1)):].strip()
    
    # RAM usage
    ram_match = re.search(r'RAM (\d+)/(\d+)MB \(lfb (\d+)x(\d+)MB\)', line)
    if ram_match:
        data['ram_used_mb'] = int(ram_match.group(1))
        data['ram_total_mb'] = int(ram_match.group(2))
        data['ram_lfb_blocks'] = int(ram_match.group(3))
        data['ram_lfb_size_mb'] = int(ram_match.group(4))
    
    # SWAP usage
    swap_match = re.search(r'SWAP (\d+)/(\d+)MB \(cached (\d+)MB\)', line)
    if swap_match:
        data['swap_used_mb'] = int(swap_match.group(1))
        data['swap_total_mb'] = int(swap_match.group(2))
        data['swap_cached_mb'] = int(swap_match.group(3))
    
    # IRAM usage
    iram_match = re.search(r'IRAM (\d+)/(\d+)kB(?:\(lfb (\d+)kB\))?', line)
    if iram_match:
        data['iram_used_kb'] = int(iram_match.group(1))
        data['iram_total_kb'] = int(iram_match.group(2))
        if iram_match.group(3):
            data['iram_lfb_kb'] = int(iram_match.group(3))
    
    # CPU cores - handle variable frequencies per core
    cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
    if cpu_match:
        cpu_data = cpu_match.group(1)
        cores = cpu_data.split(',')
        for i, core_info in enumerate(cores):
            core_info = core_info.strip()
            if '%@' in core_info:
                parts = core_info.split('%@')
                if len(parts) == 2:
                    usage = int(parts[0])
                    freq = int(parts[1])
                    data[f'cpu{i}_usage_pct'] = usage
                    data[f'cpu{i}_freq_mhz'] = freq
            elif core_info.endswith('%'):
                usage = int(core_info[:-1])
                data[f'cpu{i}_usage_pct'] = usage
    
    # EMC (External Memory Controller)
    emc_match = re.search(r'EMC (\d+)%@(\d+)', line)
    if emc_match:
        data['emc_usage_pct'] = int(emc_match.group(1))
        data['emc_freq_mhz'] = int(emc_match.group(2))
    
    # GPU (GR3D_FREQ) - handle both single and dual GPC formats
    gpu_dual_match = re.search(r'GR3D_FREQ (\d+)%@\[(\d+),(\d+)\]', line)
    gpu_single_match = re.search(r'GR3D_FREQ (\d+)%@?(\d+)?', line)
    
    if gpu_dual_match:
        data['gpu_usage_pct'] = int(gpu_dual_match.group(1))
        data['gpu_gpc0_freq_mhz'] = int(gpu_dual_match.group(2))
        data['gpu_gpc1_freq_mhz'] = int(gpu_dual_match.group(3))
    elif gpu_single_match:
        data['gpu_usage_pct'] = int(gpu_single_match.group(1))
        if gpu_single_match.group(2):
            data['gpu_freq_mhz'] = int(gpu_single_match.group(2))
    
    # VIC (Video Image Compositor)
    vic_match = re.search(r'VIC_FREQ (\d+)%@(\d+)', line)
    if vic_match:
        data['vic_usage_pct'] = int(vic_match.group(1))
        data['vic_freq_mhz'] = int(vic_match.group(2))
    
    # APE (Audio Processing Engine)
    ape_match = re.search(r'APE (\d+)', line)
    if ape_match:
        data['ape_freq_mhz'] = int(ape_match.group(1))
    
    # MTS (foreground/background tasks)
    mts_match = re.search(r'MTS fg (\d+)% bg (\d+)%', line)
    if mts_match:
        data['mts_fg_pct'] = int(mts_match.group(1))
        data['mts_bg_pct'] = int(mts_match.group(2))
    
    # NVENC (Video Encoder)
    nvenc_match = re.search(r'NVENC (\d+)', line)
    if nvenc_match:
        data['nvenc_freq_mhz'] = int(nvenc_match.group(1))
    
    # NVDEC (Video Decoder)
    nvdec_match = re.search(r'NVDEC (\d+)', line)
    if nvdec_match:
        data['nvdec_freq_mhz'] = int(nvdec_match.group(1))
    
    # NVDLA (Deep Learning Accelerator)
    for nvdla_match in re.finditer(r'NVDLA(\d+) (\d+)%@(\d+)|NVDLA(\d+) (\d+)', line):
        if nvdla_match.group(1):  # Format: NVDLA0 X%@Y
            nvdla_id = nvdla_match.group(1)
            usage = nvdla_match.group(2)
            freq = nvdla_match.group(3)
            data[f'nvdla{nvdla_id}_usage_pct'] = int(usage)
            data[f'nvdla{nvdla_id}_freq_mhz'] = int(freq)
        elif nvdla_match.group(4):  # Format: NVDLA0 Y
            nvdla_id = nvdla_match.group(4)
            freq = nvdla_match.group(5)
            data[f'nvdla{nvdla_id}_freq_mhz'] = int(freq)
    
    # GR3D_PCI (DGPU)
    gr3d_pci_match = re.search(r'GR3D_PCI (\d+)%@(\d+)|GR3D_PCI (\d+)%|GR3D_PCI (\d+)', line)
    if gr3d_pci_match:
        if gr3d_pci_match.group(1):
            data['dgpu_usage_pct'] = int(gr3d_pci_match.group(1))
            data['dgpu_freq_mhz'] = int(gr3d_pci_match.group(2))
        elif gr3d_pci_match.group(3):
            data['dgpu_usage_pct'] = int(gr3d_pci_match.group(3))
        elif gr3d_pci_match.group(4):
            data['dgpu_freq_mhz'] = int(gr3d_pci_match.group(4))
    
    # Temperature sensors
    for temp_match in re.finditer(r'([a-zA-Z0-9_]+)@([0-9.]+)C', line):
        sensor_name = temp_match.group(1).lower()
        temp_c = float(temp_match.group(2))
        data[f'temp_{sensor_name}_c'] = temp_c
    
    # Power rails - more comprehensive pattern matching
    for power_match in re.finditer(r'([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW', line):
        rail_name = power_match.group(1).lower()
        current_mw = int(power_match.group(2))
        avg_mw = int(power_match.group(3))
        data[f'{rail_name}_current_mw'] = current_mw
        data[f'{rail_name}_avg_mw'] = avg_mw
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Parse tegrastats log into comprehensive CSV')
    parser.add_argument('--input', '-i', default='tegrastats.log', help='Input tegrastats log file')
    parser.add_argument('--output', '-o', default='energy.csv', help='Output CSV file')
    args = parser.parse_args()

    rows = []
    with open(args.input, 'r') as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            row = parse_line(line)
            if row:
                row['line_number'] = lineno
                rows.append(row)

    if not rows:
        print("No data parsed from input file.")
        return

    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    
    # Sort fieldnames for consistent output
    fieldnames = sorted(all_fieldnames)

    with open(args.output, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Parsed {len(rows)} lines into {args.output}")
    print(f"Extracted {len(fieldnames)} different metrics:")
    # for field in fieldnames[:10]:
    #     print(f"  {field}")
    # if len(fieldnames) > 10:
    #     print(f"  ... and {len(fieldnames) - 10} more")

if __name__ == '__main__':
    main()