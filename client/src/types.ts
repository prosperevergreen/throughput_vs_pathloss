export interface CSVFileType {
  '1. best RSRP': string;
  BI: string;
  BT: string;
  Band: string;
  'Band (MHz)': string;
  Height: string | number;
  'NR PCI Beam index': string | number;
  'NR-ARFCN': string | number;
  PCI: string | number;
  'PDCP DL bitrate': string | number;
  Time: string | number;
  _oid: string | number;
}

export interface CSVProcessDataType {
  pathloss: number;
  pdcp: number;
}
