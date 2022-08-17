import axios from 'axios';
import { CSVProcessDataType } from '../types';

interface PostThroughputVsPathloss {
  oldCSVData: CSVProcessDataType[];
  newCSVData: CSVProcessDataType[];
}

export const apiPostThroughputVsPathloss = (data: PostThroughputVsPathloss) =>
  axios
    .post('http://localhost:5005/api/throughput-vs-pathloss', data)
    .then((res) => res);
