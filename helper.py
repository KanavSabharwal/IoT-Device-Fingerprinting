import pandas as pd
import os

pd.options.mode.chained_assignment = None

def check_polarity(ip_src:str)->int:
    """Check the direction of a packet

    Args:
        ip_src (str): Source IP Address

    Returns:
        int: -1 for incoming packets, 1 for outgoing packets
    """
    if ip_src.startswith('192'):
        return 1
    else:
        return -1

def split_traces_mseq():
    days_csv = os.listdir('data/original-trace')
    days_csv = [file for file in days_csv if not (file.startswith('.') or file =='placeholder.txt')]
    
    global_pcaps_arr = []
    for day in days_csv:
        print("Reading", day)
        global_pcap = pd.read_csv('data/original-trace/' + day, delimiter='\t',low_memory=False)

        # filter only outgoing connections (packets an adversary can see)
        relevant_pcap = global_pcap[
            (global_pcap['ip.src'].str.contains("^192.*$", na=False)
                & ~global_pcap['ip.dst'].str.contains("^192.*$", na=False))
            | (~global_pcap['ip.src'].str.contains("^192.*$", na=False)
                & global_pcap['ip.dst'].str.contains("^192.*$", na=False))]

        # filter out non TCP/UDP packets
        relevant_pcap['ip.proto'] = pd.to_numeric(relevant_pcap['ip.proto'], errors='coerce', downcast='integer')
        relevant_pcap = relevant_pcap[(relevant_pcap['ip.proto'] == 6) | (relevant_pcap['ip.proto'] == 17)]
        relevant_pcap['tcp.len'] = relevant_pcap['tcp.len'].fillna(-1)
        relevant_pcap['udp.length'] = relevant_pcap['udp.length'].fillna(-1)

        # correct UDP length (original capture contains the entire packet len)
        relevant_pcap.loc[relevant_pcap['udp.length'] != -1, 'udp.length'] = relevant_pcap['udp.length'] - 8

        # copy tcp.len or udp.length to transport.len
        relevant_pcap['transport.len'] = relevant_pcap.apply(lambda x: int(x['udp.length']) if x['ip.proto'] == 17 else x['tcp.len'], axis=1)
        global_pcaps_arr.append(relevant_pcap)

    # combined_pcaps = pd.concat([df for df in global_pcaps_arr], ignore_index=True, sort=False)

    devices_list = pd.read_csv('data/devices.csv', delimiter='\t')


    # create csvs by device and day
    op_basepath = 'data/split-trace/'
    day_idx = 0
    for day_pcap in global_pcaps_arr:
        # remove tcp keep alives
        no_tcp_alive_day_pcap = day_pcap[day_pcap['transport.len'] != 0.0]
        # add device IP column (if src beginswith 192 then IP else dst is IP)
        no_tcp_alive_day_pcap['dev.ip'] = no_tcp_alive_day_pcap.apply(lambda x: x['ip.src'] if x['ip.src'].startswith('192.') else x['ip.dst'], axis=1)
        for dev_ip, dev_pcap in day_pcap.groupby(no_tcp_alive_day_pcap['dev.ip']):
            try:
                devname = devices_list[devices_list['IPs'].str.contains(dev_ip)]['Hostname'].iloc[0]
                dev_pcap.to_csv(op_basepath + "day_" + str(day_idx) + '_' + devname + '.csv', index=False)
            except:
                pass

        day_idx+=1