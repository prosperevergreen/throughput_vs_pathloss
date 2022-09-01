import axios from 'axios';
import { CSVProcessDataType } from '../types';

interface PostThroughputVsPathloss {
  oldCSVData: CSVProcessDataType[];
  newCSVData: CSVProcessDataType[];
}

export const apiPostThroughputVsPathloss = (data: PostThroughputVsPathloss) =>
  axios
    .post(`http://${window.location.hostname}:5005/api/throughput-vs-pathloss`, data, {responseType: 'arraybuffer'})
    .then((res) => res);
